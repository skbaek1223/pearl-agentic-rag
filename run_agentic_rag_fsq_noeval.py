from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import spacy
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SEARCH_O1_WIKI_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_o1_baselines")
sys.path.insert(0, SEARCH_O1_WIKI_SCRIPTS)
from retriever_utils import RemoteRetriever
from evaluate import run_evaluation

sys.path.insert(0, str(Path(__file__).parent))
from modernbert_evaluator import ModernBertEvaluator, RemoteModernBertEvaluator, DEFAULT_CHECKPOINT as DEFAULT_EVALUATOR_CHECKPOINT
from planner_infer import Planner, DEFAULT_ADAPTER_DIR as DEFAULT_PLANNER_ADAPTER
from local_prompts_fsq_noeval import (
    get_single_qa_forced_search_no_evaluator_instruction, get_multi_qa_forced_search_no_evaluator_instruction,
)
from local_prompts import (
    get_single_qa_instruction, get_multi_qa_instruction, get_extractor_instruction,
    get_single_qa_no_planner_instruction, get_multi_qa_no_planner_instruction,
    get_single_qa_no_planner_no_evaluator_instruction, get_multi_qa_no_planner_no_evaluator_instruction,
    get_single_qa_no_evaluator_instruction, get_multi_qa_no_evaluator_instruction,
    get_single_qa_forced_search_instruction, get_multi_qa_forced_search_instruction,
    get_qa_zero_shot_instruction,
    get_simpledeepsearcher_qa_instruction, get_simpledeepsearcher_task_instruction_openqa,
    get_simpledeepsearcher_extraction_instruction,
    strip_mc_choices,
)

MULTI_HOP_DATASETS = {"hotpotqa", "2wiki", "musique", "bamboogle"}

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
EXTRACTED_MARKER = "**Extracted Information**"
SIMPLEDEEPSEARCHER_EXTRACTED_MARKER = "**Final Information**"

_PHRASAL_PARTICLES = {"out", "up", "for", "into", "about", "off", "down"}
_NLP = spacy.load("en_core_web_sm")

_KNOWN_LEAD_VERBS = {
    "identify", "find", "retrieve", "look", "determine", "compare",
    "confirm", "search", "verify", "list", "extract", "locate", "select",
    "check", "match", "lookup", "conclude", "infer", "return", "query",
    "synthesize", "use", "filter", "provide", "combine",
}


def make_trigger_sentence(steps: list[str]) -> str:
    if not steps:
        return ""
    clauses = []
    for i, s in enumerate(steps):
        s_low = s[0].lower() + s[1:] if s else s
        clauses.append(f"let's {s_low.rstrip('.')}" if i > 0 else f"Let's {s_low.rstrip('.')}")
    return ", then ".join(clauses) + "."


def make_reason_trigger_sentence(stripped_step0: str, remaining_steps: list[str]) -> str:
    clauses = [f"Let's work out {stripped_step0.rstrip('.')}"]
    for s in remaining_steps:
        s_low = s[0].lower() + s[1:] if s else s
        clauses.append(f"let's {s_low.rstrip('.')}")
    return ", then ".join(clauses) + "."


def strip_leading_verb_from_doc(step_text: str, doc) -> str:
    if len(doc) == 0:
        return step_text
    root = doc[0]
    is_parsed_verb = root.pos_ in ("VERB", "AUX")
    is_known_verb = root.text.lower() in _KNOWN_LEAD_VERBS
    if not (is_parsed_verb or is_known_verb):
        return step_text
    end_idx = root.i + 1
    if end_idx < len(doc):
        nxt = doc[end_idx]
        is_prt = any(c.dep_ == "prt" and c.i == end_idx for c in root.children)
        is_particle_word = nxt.text.lower() in _PHRASAL_PARTICLES and nxt.dep_ in ("prep", "prt")
        if is_prt or is_particle_word:
            end_idx += 1
    rest = doc[end_idx:].text.strip()
    return rest if rest else step_text


_FILLER_TAIL_RE = re.compile(
    r"\b(as described (in|by)|from a reliable\b[^,.]*|according to\b[^,.]*|"
    r"to (confirm|ensure|verify|determine)\b.*)",
    re.IGNORECASE,
)
_QUOTED_SPAN_RE = re.compile(r"'[^']+'|\"[^\"]+\"")
_KEYWORD_POS = {"NOUN", "PROPN", "NUM", "ADJ", "VERB"}


def compress_to_keywords(step_text: str) -> str:
    text = _FILLER_TAIL_RE.split(step_text)[0]
    quotes: list[str] = []

    def _stash(m: re.Match) -> str:
        quotes.append(m.group(0))
        return f" QUOTEPLACEHOLDER{len(quotes) - 1} "

    stashed = _QUOTED_SPAN_RE.sub(_stash, text)
    doc = _NLP(stashed)
    tokens = []
    for tok in doc:
        if tok.text.upper().startswith("QUOTEPLACEHOLDER"):
            tokens.append(tok.text)
        elif tok.pos_ in _KEYWORD_POS:
            tokens.append(tok.text)
    out = " ".join(tokens)
    for i, q in enumerate(quotes):
        out = out.replace(f"QUOTEPLACEHOLDER{i}", q)
    return out if out.strip() else step_text


def extract_between(text: str, start_tag: str, end_tag: str) -> str | None:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def has_complete_boxed(text: str) -> bool:
    match = re.search(r"\\boxed\{", text)
    if not match:
        return False
    depth = 1
    for i in range(match.end(), len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        if depth == 0:
            return True
    return False


def truncate_after_boxed(text: str) -> str:
    match = re.search(r"\\boxed\{", text)
    if not match:
        return text
    depth = 1
    for i in range(match.end(), len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        if depth == 0:
            return text[: i + 1]
    return text


def format_docs(docs: list[dict]) -> str:
    return "".join(
        f"**Doc {i + 1}:**\n{json.dumps(d, ensure_ascii=False, indent=2)}\n"
        for i, d in enumerate(docs)
    )


CALCULATION_HEAVY_DATASETS = {"gpqa", "hle"}


def eval_guide_text(question: str, is_suff: bool,
                     dataset_name: str | None = None) -> str:
    if is_suff:
        return (
            f"Sufficient.\n"
            f"1) If there is still more information to retrieve before fully answering the "
            f"original question \"{question}\", derive an intermediate answer based on the "
            f"retrieved information, then continue reasoning toward the next retrieval. You "
            f"may refer to the reasoning context above or the retrieval guide to inform your "
            f"search strategy, or feel free to take a different approach.\n"
            f"2) If you have sufficient information to fully answer the original question, "
            f"provide the final answer in the format \\boxed{{YOUR_ANSWER}}."
        )
    if dataset_name in CALCULATION_HEAVY_DATASETS:
        return (
            f"Insufficient.\n"
            f"Feel free to explore alternative paths, such as trying a different search query "
            f"or taking other retrieval steps as needed -- but if what you actually need right "
            f"now is a calculation or derivation you can work out yourself rather than "
            f"something to look up, do that instead of searching again; you can also work out "
            f"an intermediate result yourself and then search for whatever specific fact that "
            f"result now points you toward, to derive an answer to the question: {question}."
        )
    return (
        f"Insufficient.\n"
        f"Feel free to explore alternative paths, such as trying a different search query or "
        f"taking other retrieval steps as needed, to derive an answer to the question: {question}."
    )


def _classify_step1_types(planner, questions: list[str], plans: list[list[str]],
                           dataset_name: str, stats: dict) -> list[str]:
    if dataset_name not in CALCULATION_HEAVY_DATASETS:
        return ["lookup"] * len(plans)
    idx_with_step1 = [i for i, p in enumerate(plans) if p]
    labels = ["lookup"] * len(plans)
    if not idx_with_step1:
        return labels
    step1_raw = [plans[i][0] for i in idx_with_step1]
    docs = list(_NLP.pipe(step1_raw))
    step1_stripped = [strip_leading_verb_from_doc(s, d) for s, d in zip(step1_raw, docs)]
    q_for_classify = [questions[i] for i in idx_with_step1]
    types = planner.classify_step1_type(q_for_classify, step1_stripped)
    n_reason = 0
    for i, t in zip(idx_with_step1, types):
        labels[i] = t
        if t == "reason":
            n_reason += 1
    print(f"[planner] step1 type: {n_reason}/{len(idx_with_step1)} classified 'reason' "
          f"(forced search skipped), {len(idx_with_step1) - n_reason} 'lookup' (forced search as before)")
    stats["step1_type_n_reason"] = n_reason
    stats["step1_type_n_lookup"] = len(idx_with_step1) - n_reason
    return labels


def parse_args():
    p = argparse.ArgumentParser(
        description="Planner-forced-first-search + ModernBERT-evaluated agentic RAG."
    )
    p.add_argument("--dataset_name", required=True,
                   choices=["nq", "triviaqa", "popqa", "hotpotqa", "2wiki", "musique",
                            "bamboogle", "ambigqa", "webquestions"])
    p.add_argument("--split", default="test")
    p.add_argument("--qa_data_path", required=True,
                   help="QA dataset JSON: a list of {Question, answer, ...} items.")
    p.add_argument("--subset_num", type=int, default=-1)

    p.add_argument("--retriever_url", required=False, default=None,
                   help="e.g. http://127.0.0.1:8765.")
    p.add_argument("--force_multi_hop", choices=["auto", "true", "false"], default="auto",
                   help='Override the dataset-level hop type for the whole run (auto: from --dataset_name).')
    p.add_argument("--use_hop_router", action="store_true",
                   help='Route each question to single-/multi-hop mode with the Qwen3-8B hop router before planning.')
    p.add_argument("--hop_router_skip_single_planner", action="store_true",
                   help='With --use_hop_router: single-hop-routed questions skip the planner and use the question itself as the forced first query (deployed PEARL).')
    p.add_argument("--no_planner", action="store_true",
                   help='Ablation: no planner and no forced first search; the agent chooses its own first action.')
    p.add_argument("--forced_search_question_only", action="store_true",
                   help='Ablation: force the bare question as the first search query, without a planner.')
    p.add_argument("--no_evaluator", action="store_true",
                   help='Ablation: disable the sufficiency evaluator and its [Reasoning Guide] messages.')
    p.add_argument("--no_search", action="store_true",
                   help='Ablation: no retrieval (zero-shot instruction, no search calls).')
    p.add_argument("--no_extractor", action="store_true",
                   help='Ablation: show raw retrieved documents instead of the extracted fact.')
    p.add_argument("--simpledeepsearcher_prompt", action="store_true",
                   help="Use SimpleDeepSearcher's original prompts (for evaluating its checkpoint).")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_doc_len", type=int, default=3000)
    p.add_argument("--max_search_limit", type=int, default=10)
    p.add_argument("--max_turn", type=int, default=20,
                   help="Max FREE (post-forced-first-search) reasoning turns.")
    p.add_argument("--retrieval_method", default="e5", choices=["e5", "bm25"],
                   help="Tag used for the on-disk search cache filename; must match the server.")
    p.add_argument("--search_cache_suffix", default="",
                   help="Optional suffix for the search cache file (avoids races across parallel runs).")

    p.add_argument("--model_path", default=os.environ.get("PEARL_QWQ_MODEL_PATH"), required=False,
                   help='Reasoning LLM (also used for extraction). Defaults to $PEARL_QWQ_MODEL_PATH.')
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--top_k_sampling", type=int, default=20)
    p.add_argument("--repetition_penalty", type=float, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help='vLLM sampling seed.')
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--max_model_len", type=int, default=20480)
    p.add_argument("--max_output_chars", type=int, default=40_000)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90,
                    help="vLLM gpu_memory_utilization for the reasoning LLM. Lower this "
                         "when sharing a GPU with another process that already holds memory.")

    p.add_argument("--planner_adapter_dir", default=DEFAULT_PLANNER_ADAPTER,
                   help="LoRA adapter dir for the planner. Pass 'none' to use the "
                        "untrained base model as the planner (trained-vs-untrained "
                        "planner ablation).")
    p.add_argument("--planner_base_model", default="Qwen/Qwen3-8B")
    p.add_argument("--plan_cache_path", default=None,
                   help='JSON {question: [step, ...]} from precompute_plans.py; if omitted, plans are generated in-process.')

    p.add_argument("--evaluator_checkpoint", default=DEFAULT_EVALUATOR_CHECKPOINT)
    p.add_argument("--evaluator_device", default="cuda:0")
    p.add_argument("--evaluator_url", default=None,
                   help='URL of evaluator_server.py (e.g. http://127.0.0.1:8767); if omitted, the evaluator is loaded in-process.')

    p.add_argument("--output_dir", default=None)
    p.add_argument("--gpu_hourly_rate", type=float, default=None,
                   help='Optional $/GPU-hour rate for a serving-cost estimate in run_stats.json.')
    p.add_argument("--extractor_force_single_hop", action="store_true",
                   help='Use the single-hop extractor rules on a multi-hop dataset.')
    p.add_argument("--disable_final_answer_note", action="store_true",
                   help='Omit the single-hop final-answer note from the instruction.')
    p.add_argument("--disable_query_compression", dest="disable_query_compression",
                   action="store_true", default=True,
                   help='Do not keyword-compress the forced first query (default).')
    p.add_argument("--enable_query_compression", dest="disable_query_compression",
                   action="store_false",
                   help='Keyword-compress the forced first query.')
    return p.parse_args()


def main():
    args = parse_args()
    if not args.retriever_url:
        raise SystemExit("--retriever_url is required.")
    if args.no_planner and (args.plan_cache_path or args.use_hop_router):
        raise SystemExit("--no_planner is incompatible with --plan_cache_path and --use_hop_router "
                          "(there is no plan to precompute/route with when there's no planner).")
    if args.forced_search_question_only and (args.no_planner or args.plan_cache_path or args.use_hop_router):
        raise SystemExit("--forced_search_question_only is incompatible with --no_planner, "
                          "--plan_cache_path, and --use_hop_router (it supplies its own "
                          "single-step 'plan' -- the bare question -- in place of all three).")
    if args.hop_router_skip_single_planner and not args.use_hop_router:
        raise SystemExit("--hop_router_skip_single_planner requires --use_hop_router "
                          "(there is no per-question hop label to key the skip on otherwise).")
    if args.no_search and not args.no_planner:
        raise SystemExit("--no_search requires --no_planner (there is nothing for a plan to point "
                          "a first search at when no search will ever run).")
    if args.no_search:
        args.max_search_limit = 0
    if args.no_extractor and not args.no_planner:
        raise SystemExit("--no_extractor currently requires --no_planner -- the forced-round "
                          "extractor call (planner-present regimes only) isn't wired to the "
                          "raw-docs bypass yet, so a planner-present run would extract on turn 1 "
                          "and skip extraction on every later turn, an inconsistent ablation.")
    if args.simpledeepsearcher_prompt and not (args.no_planner and args.no_evaluator):
        raise SystemExit("--simpledeepsearcher_prompt requires --no_planner --no_evaluator "
                          "(SimpleDeepSearcher's own inference.py has no planner or trained "
                          "sufficiency evaluator either).")
    if args.repetition_penalty is None:
        args.repetition_penalty = 1.05 if "qwq" in args.model_path.lower() else 1.0
    if args.force_multi_hop == "auto":
        is_multi_hop = args.dataset_name in MULTI_HOP_DATASETS
    else:
        is_multi_hop = args.force_multi_hop == "true"
    extractor_is_multi_hop = is_multi_hop and not args.extractor_force_single_hop

    with open(args.qa_data_path, encoding="utf-8") as f:
        qa_data = json.load(f)
    filtered_data = list(qa_data)
    if args.subset_num != -1:
        filtered_data = filtered_data[: args.subset_num]
    questions = [item["Question"] for item in filtered_data]
    print(f"Loaded {len(filtered_data)} questions from {args.qa_data_path}")
    planner_questions = [strip_mc_choices(q) for q in questions]

    model_short = args.model_path.split("/")[-1].lower().replace("-instruct", "")
    output_dir = args.output_dir or f"./outputs/{args.dataset_name}.{model_short}.planner_modernbert_rag"
    os.makedirs(output_dir, exist_ok=True)

    live_traj_path = os.path.join(output_dir, "trajectories_live.jsonl")
    live_traj_file = open(live_traj_path, "w", encoding="utf-8")

    def write_trajectory_live(seq: dict):
        record = {
            "id": seq["item"].get("id"),
            "question": seq["question"],
            "output": seq["output"],
            "finish_reason": seq.get("finish_reason", "unresolved"),
            "search_count": seq["search_count"],
        }
        live_traj_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        live_traj_file.flush()

    retriever = RemoteRetriever(args.retriever_url, max_doc_len=args.max_doc_len)
    print(f"[retrieval] mode=wiki ({args.retriever_url})")

    stats = {
        "n_questions": len(filtered_data),
        "planner_prompt_tokens": 0, "planner_completion_tokens": 0,
        "reasoning_prompt_tokens": 0, "reasoning_completion_tokens": 0,
        "extractor_prompt_tokens": 0, "extractor_completion_tokens": 0,
        "retrieval_queries_issued": 0,
        "retrieval_server_calls": 0,
        "n_turns_total": 0,
        "wall_time_planner_s": 0.0,
        "wall_time_evaluator_load_s": 0.0,
        "wall_time_reasoning_llm_load_s": 0.0,
        "wall_time_generation_s": 0.0,
        "wall_time_total_s": 0.0,
    }

    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    _cache_tag = args.retrieval_method
    if args.search_cache_suffix:
        _cache_tag = f"{_cache_tag}.{args.search_cache_suffix}"
    search_cache_path = os.path.join(cache_dir, f"search_cache.{_cache_tag}.json")
    search_cache = json.load(open(search_cache_path, encoding="utf-8")) if os.path.exists(search_cache_path) else {}

    def save_search_cache():
        with open(search_cache_path, "w", encoding="utf-8") as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)

    def cached_batch_search(queries: list[str]) -> dict[str, list[dict]]:
        stats["retrieval_queries_issued"] += len(queries)
        fresh = [q for q in dict.fromkeys(queries) if q not in search_cache]
        if fresh:
            stats["retrieval_server_calls"] += len(fresh)
            try:
                docs = retriever.batch_search(fresh, args.top_k)
            except Exception as e:
                print(f"Retrieval batch error: {e}")
                docs = [[] for _ in fresh]
            for q, d in zip(fresh, docs):
                search_cache[q] = d
        return {q: search_cache[q] for q in queries}

    t0 = time.time()
    if args.no_planner:
        print("[planner] --no_planner: skipping planner stage entirely (no model loaded)")
        plans = [[] for _ in questions]
        per_item_is_multi_hop = [is_multi_hop] * len(questions)
        per_item_step1_type = ["lookup"] * len(questions)
        per_item_planner_skipped = [False] * len(questions)
    elif args.forced_search_question_only:
        print("[planner] --forced_search_question_only: no Planner model loaded; "
              "forcing the bare question as each item's single-step 'plan'")
        plans = [[q] for q in questions]
        per_item_is_multi_hop = [is_multi_hop] * len(questions)
        per_item_step1_type = ["lookup"] * len(questions)
        per_item_planner_skipped = [True] * len(questions)
    elif args.plan_cache_path:
        print(f"[planner] loading precomputed plans from {args.plan_cache_path}...")
        plan_cache = json.load(open(args.plan_cache_path, encoding="utf-8"))
        missing = [q for q in questions if q not in plan_cache]
        if missing:
            raise SystemExit(
                f"[planner] {len(missing)}/{len(questions)} questions missing from "
                f"{args.plan_cache_path} (e.g. {missing[0]!r}) -- run precompute_plans.py "
                f"on this qa_data_path first."
            )
        plans = [plan_cache[q] for q in questions]
        print(f"[planner] loaded {len(plans)} plans from cache")
        per_item_is_multi_hop = [is_multi_hop] * len(questions)
        per_item_step1_type = ["lookup"] * len(questions)
        per_item_planner_skipped = [False] * len(questions)
    elif args.use_hop_router:
        if not args.planner_adapter_dir or str(args.planner_adapter_dir).lower() == "none":
            raise SystemExit("--use_hop_router requires --planner_adapter_dir to be a "
                              "real trained adapter (there is nothing to route TO otherwise).")
        print(f"[planner] loading adapter {args.planner_adapter_dir} on base {args.planner_base_model} "
              f"(hop-router mode: classify then route per question"
              + (", single-hop skips planner entirely)..." if args.hop_router_skip_single_planner else ")..."))
        planner = Planner(args.planner_adapter_dir, args.planner_base_model)
        plans, hop_labels = planner.route_and_generate_plans(
            planner_questions, dataset_name=args.dataset_name,
            skip_planner_for_single=args.hop_router_skip_single_planner)
        per_item_is_multi_hop = [label == "multi" for label in hop_labels]
        per_item_planner_skipped = [args.hop_router_skip_single_planner and label == "single"
                                     for label in hop_labels]
        n_multi = sum(per_item_is_multi_hop)
        print(f"[planner] hop router: {n_multi}/{len(questions)} routed multi "
              f"(trained adapter), {len(questions) - n_multi} routed single "
              f"({'planner skipped, bare question forced search' if args.hop_router_skip_single_planner else 'untrained planner'})")
        stats["hop_router_n_multi"] = n_multi
        stats["hop_router_n_single"] = len(questions) - n_multi
        stats["planner_prompt_tokens"] = planner.last_prompt_tokens
        stats["planner_completion_tokens"] = planner.last_completion_tokens
        per_item_step1_type = _classify_step1_types(planner, planner_questions, plans, args.dataset_name, stats)
        print(f"[planner] generated {len(plans)} plans; freeing planner from GPU memory")
        del planner
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"[planner] loading adapter {args.planner_adapter_dir} on base {args.planner_base_model}...")
        planner = Planner(args.planner_adapter_dir, args.planner_base_model)
        plans = planner.generate_plans(planner_questions, is_multi_hop=is_multi_hop, dataset_name=args.dataset_name)
        per_item_is_multi_hop = [is_multi_hop] * len(questions)
        per_item_planner_skipped = [False] * len(questions)
        stats["planner_prompt_tokens"] = planner.last_prompt_tokens
        stats["planner_completion_tokens"] = planner.last_completion_tokens
        per_item_step1_type = _classify_step1_types(planner, planner_questions, plans, args.dataset_name, stats)
        print(f"[planner] generated {len(plans)} plans; freeing planner from GPU memory")
        del planner
        gc.collect()
        torch.cuda.empty_cache()
    stats["wall_time_planner_s"] = time.time() - t0

    t0 = time.time()
    if args.no_evaluator:
        print("[evaluator] --no_evaluator: skipping evaluator entirely (no model loaded)")
        evaluator = None
    elif args.evaluator_url:
        print(f"[evaluator] using remote evaluator at {args.evaluator_url} "
              f"(no local GPU memory used)")
        evaluator = RemoteModernBertEvaluator(args.evaluator_url)
    else:
        print(f"[evaluator] loading ModernBERT checkpoint {args.evaluator_checkpoint} "
              f"on {args.evaluator_device}...")
        evaluator = ModernBertEvaluator(args.evaluator_checkpoint, device=args.evaluator_device)
    stats["wall_time_evaluator_load_s"] = time.time() - t0

    t0 = time.time()
    print(f"[reasoning] loading {args.model_path} via vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    n_gpus = len([g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]) or 1
    llm = LLM(model=args.model_path, tensor_parallel_size=n_gpus, gpu_memory_utilization=args.gpu_memory_utilization,
              dtype="half", max_model_len=args.max_model_len, max_num_seqs=256,
              disable_custom_all_reduce=True)
    stats["n_gpus_reasoning_llm"] = n_gpus
    stats["wall_time_reasoning_llm_load_s"] = time.time() - t0

    MAX_MODEL_LEN = args.max_model_len
    MIN_OUTPUT_ROOM = 512

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    extractor_sampling = SamplingParams(max_tokens=4096, temperature=0.0, top_p=1.0)

    def make_main_sampling(max_new: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_new, temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k_sampling, repetition_penalty=args.repetition_penalty,
            stop=[END_SEARCH_QUERY], include_stop_str_in_output=True,
            seed=args.seed,
        )

    def llm_chat_batch(user_prompts: list[str], sp: SamplingParams, category: str = "other") -> list[str]:
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": up}],
                                           tokenize=False, add_generation_prompt=True)
            for up in user_prompts
        ]
        outs = llm.generate(prompts, sampling_params=sp)
        stats[f"{category}_prompt_tokens"] += sum(len(o.prompt_token_ids) for o in outs)
        stats[f"{category}_completion_tokens"] += sum(len(o.outputs[0].token_ids) for o in outs)
        return [o.outputs[0].text for o in outs]

    def run_extractor(items: list[tuple[str, str, str, str, bool]]) -> list[str]:
        if args.simpledeepsearcher_prompt:
            prompts = [
                get_simpledeepsearcher_extraction_instruction(prev_reasoning=rr, search_query=sq, document=docs)
                for q, rr, sq, docs, item_extractor_is_multi_hop in items
            ]
        else:
            prompts = [
                get_extractor_instruction(question=q, recent_reasoning=rr, search_query=sq, documents=docs,
                                           is_multi_hop=item_extractor_is_multi_hop, dataset_name=args.dataset_name)
                for q, rr, sq, docs, item_extractor_is_multi_hop in items
            ]
        raw = llm_chat_batch(prompts, extractor_sampling, category="extractor")
        marker = SIMPLEDEEPSEARCHER_EXTRACTED_MARKER if args.simpledeepsearcher_prompt else EXTRACTED_MARKER
        facts = []
        for r in raw:
            if marker in r:
                f = r.split(marker)[-1].strip().strip("`").strip()
                facts.append(f if f else "NONE")
            else:
                facts.append("NONE")
        return facts

    def evaluate_states(states: list[dict]) -> list[tuple[bool, float]]:
        return evaluator.predict_batch(states)

    active_sequences = []
    for item, question, steps, seq_is_multi_hop, seq_step1_type, seq_planner_skipped in zip(
            filtered_data, questions, plans, per_item_is_multi_hop, per_item_step1_type,
            per_item_planner_skipped):
        if args.simpledeepsearcher_prompt:
            instruction = get_simpledeepsearcher_qa_instruction(args.max_search_limit)
        elif args.no_search:
            instruction = get_qa_zero_shot_instruction(include_final_answer_note=not args.disable_final_answer_note)
        elif args.no_planner and args.no_evaluator:
            instruction = (get_multi_qa_no_planner_no_evaluator_instruction() if seq_is_multi_hop
                            else get_single_qa_no_planner_no_evaluator_instruction(include_final_answer_note=not args.disable_final_answer_note))
        elif args.no_planner:
            instruction = (get_multi_qa_no_planner_instruction() if seq_is_multi_hop
                            else get_single_qa_no_planner_instruction(include_final_answer_note=not args.disable_final_answer_note))
        elif args.no_evaluator and (args.forced_search_question_only or seq_planner_skipped):
            instruction = (get_multi_qa_forced_search_no_evaluator_instruction() if seq_is_multi_hop
                            else get_single_qa_forced_search_no_evaluator_instruction(include_final_answer_note=not args.disable_final_answer_note))
        elif args.no_evaluator:
            instruction = (get_multi_qa_no_evaluator_instruction() if seq_is_multi_hop
                            else get_single_qa_no_evaluator_instruction(include_final_answer_note=not args.disable_final_answer_note))
        elif args.forced_search_question_only or seq_planner_skipped:
            instruction = (get_multi_qa_forced_search_instruction() if seq_is_multi_hop
                            else get_single_qa_forced_search_instruction(include_final_answer_note=not args.disable_final_answer_note))
        else:
            instruction = get_multi_qa_instruction() if seq_is_multi_hop else get_single_qa_instruction(include_final_answer_note=not args.disable_final_answer_note)
        if args.simpledeepsearcher_prompt:
            user_prompt = get_simpledeepsearcher_task_instruction_openqa(question)
        else:
            user_prompt = (
                f"Question:\n{question}\n\n"
                + ("Begin reasoning, then give your final answer." if args.no_search
                   else "Begin reasoning, performing searches by writing search queries as needed.")
            )
        base_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction + "\n" + user_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        active_sequences.append({
            "item": item, "question": question, "steps": steps,
            "is_multi_hop": seq_is_multi_hop,
            "planner_skipped": seq_planner_skipped,
            "step1_type": seq_step1_type,
            "extractor_is_multi_hop": seq_is_multi_hop and not args.extractor_force_single_hop,
            "base_prompt": base_prompt,
            "committed_steps": [],
            "current_step": "",
            "output": "",
            "finished": False,
            "search_count": 0,
            "executed_search_queries": set(),
            "last_model_chunk": "",
            "pinned_steps": set(),
            "pin_next_step": False,
        })

    RECENT_K = 5
    ELLIPSIS = "\n\n[... omitted intermediate reasoning steps ...]\n\n"

    def _compute_keep(seq: dict, recent_k: int) -> set:
        committed = seq["committed_steps"]
        n = len(committed)
        keep = set()
        if n > 0:
            keep.add(0)
        keep.update(seq["pinned_steps"])
        keep.update(range(max(0, n - recent_k), n))
        return keep

    def _assemble(seq: dict, keep: set) -> str:
        committed = seq["committed_steps"]
        ordered = sorted(keep)
        parts = []
        prev = -1
        for i in ordered:
            if prev != -1 and i != prev + 1:
                parts.append(ELLIPSIS)
            parts.append(committed[i])
            prev = i
        return seq["base_prompt"] + "".join(parts) + seq["current_step"]

    def _truncate_to_fit(seq: dict, max_prompt_tokens: int) -> str:
        keep = _compute_keep(seq, recent_k=0)
        base = seq["base_prompt"]
        committed = seq["committed_steps"]
        ordered = sorted(keep)
        parts = []
        prev = -1
        for i in ordered:
            if prev != -1 and i != prev + 1:
                parts.append(ELLIPSIS)
            parts.append(committed[i])
            prev = i
        history_text = "".join(parts)
        cur = seq["current_step"]
        overhead = count_tokens(base + history_text + cur) - max_prompt_tokens
        if overhead <= 0:
            return base + history_text + cur
        sr_start = cur.rfind(BEGIN_SEARCH_RESULT)
        sr_end = cur.rfind(END_SEARCH_RESULT)
        if sr_start != -1 and sr_end != -1 and sr_end > sr_start:
            sr_content = cur[sr_start + len(BEGIN_SEARCH_RESULT):sr_end]
            sr_ids = tokenizer(sr_content, add_special_tokens=False)["input_ids"]
            keep_ids = sr_ids[:max(0, len(sr_ids) - overhead)]
            trimmed_sr = tokenizer.decode(keep_ids, skip_special_tokens=True)
            cur = cur[:sr_start + len(BEGIN_SEARCH_RESULT)] + trimmed_sr + cur[sr_end:]
            result = base + history_text + cur
            if count_tokens(result) <= max_prompt_tokens:
                print(f"  [truncate] trimmed search result by ~{overhead} tokens")
                return result
        cur_ids = tokenizer(cur, add_special_tokens=False)["input_ids"]
        avail = max_prompt_tokens - count_tokens(base + history_text)
        if avail > 0:
            cur = tokenizer.decode(cur_ids[-avail:], skip_special_tokens=True)
            print(f"  [truncate] hard-truncated current_step to {avail} tokens")
        else:
            cur = ""
            print(f"  [truncate] dropped current_step entirely")
        return base + history_text + cur

    def build_prompt(seq: dict, max_prompt_tokens: int) -> str:
        for k in range(RECENT_K, -1, -1):
            keep = _compute_keep(seq, recent_k=k)
            prompt = _assemble(seq, keep)
            if count_tokens(prompt) <= max_prompt_tokens:
                if k < RECENT_K:
                    print(f"  [truncate] reduced recent_k {RECENT_K} -> {k} to fit budget")
                return prompt
        return _truncate_to_fit(seq, max_prompt_tokens)

    def commit_step(seq: dict):
        if not seq["current_step"]:
            return
        idx = len(seq["committed_steps"])
        seq["committed_steps"].append(seq["current_step"])
        seq["current_step"] = ""
        if seq["pin_next_step"]:
            seq["pinned_steps"].add(idx)
            seq["pin_next_step"] = False

    start_time = time.time()

    if not args.no_planner:
        print(f"\n----- Forced round (Planner step 1) | {len(active_sequences)} sequences -----")
        for seq in active_sequences:
            steps = seq["steps"]
            if steps and (args.forced_search_question_only or seq["planner_skipped"]):
                seq["_plan_trigger"] = ""
                seq["_plan_trigger_step1"] = ""
                seq["_stripped_step0"] = steps[0]
                seq["_plan_query"] = steps[0]
            elif steps:
                doc = list(_NLP.pipe([steps[0]]))[0]
                seq["_plan_trigger"] = make_trigger_sentence(steps)
                seq["_plan_trigger_step1"] = make_trigger_sentence(steps[:1])
                stripped_step0 = strip_leading_verb_from_doc(steps[0], doc)
                seq["_stripped_step0"] = stripped_step0
                seq["_plan_query"] = stripped_step0 if args.disable_query_compression else compress_to_keywords(stripped_step0)
            else:
                seq["_plan_trigger"] = None
                seq["_plan_trigger_step1"] = None
                seq["_stripped_step0"] = None
                seq["_plan_query"] = None

        reason_seqs = [s for s in active_sequences
                       if s["_plan_query"] and s["step1_type"] == "reason"]
        for seq in reason_seqs:
            remaining = seq["steps"][1:] if seq["is_multi_hop"] else []
            trigger_for_model = make_reason_trigger_sentence(seq["_stripped_step0"], remaining)
            msg = f"{trigger_for_model}\n"
            seq["current_step"] += msg
            seq["output"] += msg

        forced_seqs = [s for s in active_sequences
                       if s["_plan_query"] and s["step1_type"] != "reason"]
        plan_queries = [s["_plan_query"] for s in forced_seqs]
        if plan_queries:
            print(f"[retrieval] forced-round queries: {len(set(plan_queries))}")
            docs_by_query = cached_batch_search(plan_queries)
        else:
            docs_by_query = {}

        extractor_items = [
            (s["question"], s["_plan_trigger_step1"], s["_plan_query"],
             format_docs(docs_by_query.get(s["_plan_query"], [])), s["extractor_is_multi_hop"])
            for s in forced_seqs
        ]
        facts = run_extractor(extractor_items) if extractor_items else []

        if args.no_evaluator:
            verdicts = [(False, 1.0)] * len(extractor_items)
        else:
            eval_states = [
                {"question": q, "recent_reasoning": rr, "search_query": sq, "extracted_info": fact}
                for (q, rr, sq, _, _), fact in zip(extractor_items, facts)
            ]
            verdicts = evaluate_states(eval_states)

        for seq, (question, _rr, q1, _, _), fact, (is_suff, conf) in zip(
                forced_seqs, extractor_items, facts, verdicts):
            if fact == "NONE" or not fact.strip():
                fact = "NONE"
                is_suff, conf = False, 1.0
            if not args.no_evaluator and is_suff and seq["is_multi_hop"]:
                seq["pinned_steps"].add(len(seq["committed_steps"]))
                seq["pin_next_step"] = True
            trigger_for_model = seq['_plan_trigger'] if seq['is_multi_hop'] else seq['_plan_trigger_step1']
            if args.no_evaluator:
                msg = (
                    f"{trigger_for_model}\n"
                    f"{BEGIN_SEARCH_QUERY}{q1}{END_SEARCH_QUERY}\n"
                    f"{BEGIN_SEARCH_RESULT}\n{EXTRACTED_MARKER}\n\n{fact}\n{END_SEARCH_RESULT}\n"
                )
            else:
                guide_text = eval_guide_text(question, is_suff, args.dataset_name)
                msg = (
                    f"{trigger_for_model}\n"
                    f"{BEGIN_SEARCH_QUERY}{q1}{END_SEARCH_QUERY}\n"
                    f"{BEGIN_SEARCH_RESULT}\n{EXTRACTED_MARKER}\n\n{fact}\n{END_SEARCH_RESULT}\n"
                    f"[Reasoning Guide]: {guide_text}\n"
                )
            seq["current_step"] += msg
            seq["output"] += msg
            seq["search_count"] += 1
            seq["executed_search_queries"].add(q1)
            commit_step(seq)

    turn = 0
    while True:
        pending = [s for s in active_sequences if not s["finished"]]
        if not pending:
            break
        turn += 1
        if turn > args.max_turn:
            print(f"Max turn {args.max_turn} reached.")
            for s in pending:
                s["finished"] = True
                s["finish_reason"] = "max_turn_global"
                write_trajectory_live(s)
            break
        print(f"\n----- Turn {turn} | {len(pending)} active -----")

        budget = MAX_MODEL_LEN - MIN_OUTPUT_ROOM
        safe_pending, safe_sps, safe_prompts = [], [], []
        for s in pending:
            built = build_prompt(s, max_prompt_tokens=budget)
            prompt_len = count_tokens(built)
            room = MAX_MODEL_LEN - prompt_len
            if room < MIN_OUTPUT_ROOM:
                msg = (f"\n{BEGIN_SEARCH_RESULT}\nContext limit reached "
                       f"(prompt {prompt_len} tokens). Terminating.\n{END_SEARCH_RESULT}\n")
                s["current_step"] += msg
                s["output"] += msg
                commit_step(s)
                s["finished"] = True
                s["finish_reason"] = "context_limit"
                write_trajectory_live(s)
                continue
            safe_pending.append(s)
            safe_sps.append(make_main_sampling(min(args.max_new_tokens, room)))
            safe_prompts.append(built)
        if not safe_pending:
            continue

        outputs = llm.generate(safe_prompts, sampling_params=safe_sps)
        stats["reasoning_prompt_tokens"] += sum(len(o.prompt_token_ids) for o in outputs)
        stats["reasoning_completion_tokens"] += sum(len(o.outputs[0].token_ids) for o in outputs)
        stats["n_turns_total"] += len(outputs)

        retrieval_candidates = []
        for seq, out in zip(safe_pending, outputs):
            text = out.outputs[0].text
            seq["last_model_chunk"] = text
            seq["current_step"] = text
            seq["output"] += text

            if len(seq["output"]) > args.max_output_chars:
                msg = (f"\n{BEGIN_SEARCH_RESULT}\nOutput length cap "
                       f"({args.max_output_chars} chars) exceeded. Terminating.\n{END_SEARCH_RESULT}\n")
                seq["current_step"] += msg
                seq["output"] += msg
                commit_step(seq)
                seq["finished"] = True
                seq["finish_reason"] = "output_length_cap"
                write_trajectory_live(seq)
                continue

            query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            if not (query and text.rstrip().endswith(END_SEARCH_QUERY)):
                if has_complete_boxed(text):
                    truncated = truncate_after_boxed(text)
                    seq["output"] = seq["output"][: len(seq["output"]) - len(text)] + truncated
                    commit_step(seq)
                    seq["finished"] = True
                    seq["finish_reason"] = "boxed_answer"
                    write_trajectory_live(seq)
                else:
                    commit_step(seq)
                continue

            if turn == args.max_turn:
                commit_step(seq)
                seq["finished"] = True
                seq["finish_reason"] = "max_turn_mid_search"
                write_trajectory_live(seq)
                continue
            if seq["search_count"] >= args.max_search_limit:
                reject_count = seq.get("search_limit_reject_count", 0)
                if reject_count >= 2:
                    msg = f"\n{BEGIN_SEARCH_RESULT}\nSearch limit reached. Terminating.\n{END_SEARCH_RESULT}\n"
                    seq["current_step"] += msg
                    seq["output"] += msg
                    commit_step(seq)
                    seq["finished"] = True
                    seq["finish_reason"] = "search_limit"
                    write_trajectory_live(seq)
                    continue
                seq["search_limit_reject_count"] = reject_count + 1
                msg = (f"\n{BEGIN_SEARCH_RESULT}\nSearch limit reached; no further "
                       f"searches are allowed.\n{END_SEARCH_RESULT}\n"
                       f"I have no searches left. Based on everything I've found so "
                       f"far, I should give my best specific answer now rather than "
                       f"say the information is insufficient.\n")
                seq["current_step"] += msg
                seq["output"] += msg
                commit_step(seq)
                continue
            if query in seq["executed_search_queries"]:
                msg = (f"\n{BEGIN_SEARCH_RESULT}\nQuery already searched. Refer to previous results."
                       f"\n{END_SEARCH_RESULT}\n"
                       f"This query was already searched. I should try a different search query "
                       f"to find new information.\n")
                seq["current_step"] += msg
                seq["output"] += msg
                commit_step(seq)
                continue

            retrieval_candidates.append((seq, query))

        if not retrieval_candidates:
            continue

        queries_needed = [q for _, q in retrieval_candidates]
        print(f"[retrieval] running {len(set(queries_needed))} unique queries "
              f"({len(queries_needed)} total)")
        docs_by_query = cached_batch_search(queries_needed)

        batch_seqs, batch_queries, batch_formatted = [], [], []
        for seq, query in retrieval_candidates:
            docs = docs_by_query.get(query, [])
            seq["search_count"] += 1
            seq["executed_search_queries"].add(query)
            batch_seqs.append(seq)
            batch_queries.append(query)
            batch_formatted.append(format_docs(docs))

        extractor_items = []
        for seq, q, docs in zip(batch_seqs, batch_queries, batch_formatted):
            last_chunk = seq.get("last_model_chunk", "")
            recent_reasoning = last_chunk.split(BEGIN_SEARCH_QUERY)[0].strip() or seq["question"]
            extractor_items.append((seq["question"], recent_reasoning, q, docs, seq["extractor_is_multi_hop"]))
        if args.no_extractor:
            facts = batch_formatted
        else:
            print(f"[extraction] extracting facts for {len(extractor_items)} queries")
            facts = run_extractor(extractor_items)

        if args.no_evaluator:
            for seq, fact in zip(batch_seqs, facts):
                if fact == "NONE" or not fact.strip():
                    fact = "NONE"
                msg = f"\n{BEGIN_SEARCH_RESULT}\n{fact}\n{END_SEARCH_RESULT}\n"
                seq["current_step"] += msg
                seq["output"] += msg
                commit_step(seq)
            continue

        eval_states = [
            {"question": q, "recent_reasoning": rr, "search_query": sq, "extracted_info": fact}
            for (q, rr, sq, _, _), fact in zip(extractor_items, facts)
        ]
        print(f"[evaluation] judging sufficiency of {len(eval_states)} extractions (ModernBERT)")
        verdicts = evaluate_states(eval_states)

        for seq, (question, rr, q, _, _), fact, (is_suff, conf) in zip(
                batch_seqs, extractor_items, facts, verdicts):
            if fact == "NONE" or not fact.strip():
                is_suff, conf, fact = False, 1.0, "NONE"
            if is_suff and seq["is_multi_hop"]:
                seq["pinned_steps"].add(len(seq["committed_steps"]))
                seq["pin_next_step"] = True
            guide_text = eval_guide_text(question, is_suff, args.dataset_name)
            msg = (f"\n{BEGIN_SEARCH_RESULT}\n{fact}\n{END_SEARCH_RESULT}\n"
                   f"[Reasoning Guide]: {guide_text}\n")
            seq["current_step"] += msg
            seq["output"] += msg
            commit_step(seq)

    total_time = time.time() - start_time
    stats["wall_time_generation_s"] = total_time
    stats["wall_time_total_s"] = (
        stats["wall_time_planner_s"] + stats["wall_time_evaluator_load_s"]
        + stats["wall_time_reasoning_llm_load_s"] + total_time
    )

    live_traj_file.close()
    input_list = [s["base_prompt"] for s in active_sequences]
    output_list = [s["output"] for s in active_sequences]
    run_evaluation(filtered_data, input_list, output_list, args.dataset_name,
                   output_dir, total_time, args.split, tokenizer=tokenizer)
    save_search_cache()

    from collections import Counter
    finish_reason_counts = Counter(s.get("finish_reason", "unresolved") for s in active_sequences)

    stats["total_reasoning_tokens"] = stats["reasoning_prompt_tokens"] + stats["reasoning_completion_tokens"]
    stats["total_extractor_tokens"] = stats["extractor_prompt_tokens"] + stats["extractor_completion_tokens"]
    stats["total_planner_tokens"] = stats["planner_prompt_tokens"] + stats["planner_completion_tokens"]
    stats["total_tokens_all"] = (stats["total_reasoning_tokens"] + stats["total_extractor_tokens"]
                                  + stats["total_planner_tokens"])

    stats["avg_turns_per_question"] = stats["n_turns_total"] / stats["n_questions"] if stats["n_questions"] else 0.0
    stats["avg_retrieval_calls_per_question"] = (
        stats["retrieval_queries_issued"] / stats["n_questions"] if stats["n_questions"] else 0.0
    )
    stats["retrieval_cache_hit_rate"] = (
        1 - stats["retrieval_server_calls"] / stats["retrieval_queries_issued"]
        if stats["retrieval_queries_issued"] else None
    )
    stats["finish_reason_counts"] = dict(finish_reason_counts)
    stats["finish_reason_fractions"] = {
        k: v / stats["n_questions"] for k, v in finish_reason_counts.items()
    } if stats["n_questions"] else {}

    gpu_seconds = (stats["wall_time_reasoning_llm_load_s"] + stats["wall_time_generation_s"]) * stats.get("n_gpus_reasoning_llm", 1)
    stats["gpu_seconds_reasoning_llm"] = gpu_seconds
    stats["gpu_hours_reasoning_llm"] = gpu_seconds / 3600
    stats["gpu_hourly_rate_used"] = args.gpu_hourly_rate
    stats["serving_cost_usd"] = (
        stats["gpu_hours_reasoning_llm"] * args.gpu_hourly_rate if args.gpu_hourly_rate is not None else None
    )
    stats["avg_latency_ms_per_question"] = (
        stats["wall_time_total_s"] / stats["n_questions"] * 1000 if stats["n_questions"] else 0.0
    )

    stats_path = os.path.join(output_dir, "run_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Done in {total_time:.1f}s. Output -> {output_dir}")
    print(f"Stats saved -> {stats_path}")
    print(f"  total_tokens_all={stats['total_tokens_all']}  "
          f"reasoning={stats['total_reasoning_tokens']}  extractor={stats['total_extractor_tokens']}  "
          f"planner={stats['total_planner_tokens']}")
    print(f"  retrieval_calls_issued={stats['retrieval_queries_issued']}  "
          f"server_calls={stats['retrieval_server_calls']}  "
          f"cache_hit_rate={stats['retrieval_cache_hit_rate']}")
    print(f"  finish_reasons={stats['finish_reason_counts']}")


if __name__ == "__main__":
    main()
