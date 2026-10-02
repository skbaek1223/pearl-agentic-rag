from __future__ import annotations

import os
import sys
from typing import Optional

SEARCH_O1_WIKI_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_o1_baselines")
QA_DATASETS = ("nq", "triviaqa", "hotpotqa", "musique", "bamboogle", "2wiki", "ambigqa", "popqa")


def last_answer(text: str) -> Optional[str]:
    end = text.rfind("</answer>")
    if end == -1:
        return None
    start = text.rfind("<answer>", 0, end)
    if start == -1:
        return None
    return text[start + len("<answer>"):end].strip()


def scoring_text(reasoning: str, pred: str) -> str:
    text = reasoning.replace("<answer>", "").replace("</answer>", "")
    return text + (f"\n\n<answer>{pred}</answer>" if pred else "")


def dataset_name_from_path(qa_data_path: str) -> str:
    base = os.path.basename(qa_data_path)
    for name in QA_DATASETS:
        if base.startswith(name + "_") or base.startswith(name + "."):
            return name
    raise ValueError(f"Cannot infer dataset name from {qa_data_path}; pass --dataset_name")


def score(qa_data: list[dict], outputs: list[str], dataset_name: str, output_dir: str,
          total_time: float, tokenizer=None, split: str = "test"):
    if SEARCH_O1_WIKI_SCRIPTS not in sys.path:
        sys.path.insert(0, SEARCH_O1_WIKI_SCRIPTS)
    from evaluate import run_evaluation
    questions = [item.get("Question") or item.get("question") for item in qa_data]
    return run_evaluation(filtered_data=qa_data, input_list=questions, output_list=outputs,
                          dataset_name=dataset_name, output_dir=output_dir, total_time=total_time,
                          split=split, apply_backoff=False, tokenizer=tokenizer)
