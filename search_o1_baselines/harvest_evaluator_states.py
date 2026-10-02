import glob
import json
import os
import re

BEGIN_Q, END_Q = "<|begin_search_query|>", "<|end_search_query|>"
BEGIN_R, END_R = "<|begin_search_result|>", "<|end_search_result|>"
GUIDE_RE = re.compile(r"\[Reasoning Guide\]: (Sufficient|Insufficient)(?: \(confidence (\d+)%\))?")

SRC_BATCHES = {
    "nq": [
        ("outputs/calib_no_retrieval_guide/nq", "../data/tune/nq_calib_hardsplit.json"),
        ("outputs/calib_no_retrieval_guide/nq_extra", "../data/tune/nq_calib_hardsplit_extra.json"),
        ("outputs/calib_no_retrieval_guide/nq_extra2", "../data/tune/nq_calib_hardsplit_extra2.json"),
        ("outputs/calib_no_retrieval_guide/nq_extra3", "../data/tune/nq_calib_hardsplit_extra3.json"),
        ("outputs/calib_no_retrieval_guide/nq_extra4", "../data/tune/nq_calib_hardsplit_extra4.json"),
    ],
    "hotpotqa": [
        ("outputs/calib_no_retrieval_guide/hotpotqa", "../data/tune/hotpotqa_calib_hardsplit.json"),
        ("outputs/calib_no_retrieval_guide/hotpotqa_extra", "../data/tune/hotpotqa_calib_hardsplit_extra.json"),
        ("outputs/calib_no_retrieval_guide/hotpotqa_extra2", "../data/tune/hotpotqa_calib_hardsplit_extra2.json"),
        ("outputs/calib_no_retrieval_guide/hotpotqa_extra3", "../data/tune/hotpotqa_calib_hardsplit_extra3.json"),
        ("outputs/calib_no_retrieval_guide/hotpotqa_extra4", "../data/tune/hotpotqa_calib_hardsplit_extra4.json"),
    ],
}
OUT_PATH = "outputs/evaluator_train/harvested_states.jsonl"


def latest_result_json(dir_path):
    candidates = [f for f in glob.glob(os.path.join(dir_path, "*.json"))
                  if "metrics" not in os.path.basename(f)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_question_lookup(qa_path):
    data = json.load(open(qa_path, encoding="utf-8"))
    return {item["id"]: item["Question"] for item in data}


def extract_states(item, dataset, question_lookup):
    out = item.get("Output", "")
    question = question_lookup.get(item.get("id"), item.get("Question", ""))
    golden_answers = item.get("golden_answers") or [item.get("answer", "")]

    queries = list(re.finditer(re.escape(BEGIN_Q) + r"(.*?)" + re.escape(END_Q), out, flags=re.DOTALL))
    results = list(re.finditer(re.escape(BEGIN_R) + r"(.*?)" + re.escape(END_R), out, flags=re.DOTALL))
    guides = list(GUIDE_RE.finditer(out))

    states = []
    for qi, q_match in enumerate(queries):
        res_after = [r for r in results if r.start() > q_match.end()]
        if not res_after:
            continue
        result_text = res_after[0].group(1).strip()
        if result_text == "NONE" or not result_text:
            continue

        prev_ends = [r.end() for r in results if r.end() < q_match.start()]
        prev_ends += [q.end() for q in queries[:qi] if q.end() < q_match.start()]
        start_ctx = max(prev_ends) if prev_ends else 0
        recent_reasoning = out[start_ctx:q_match.start()].strip()
        if not recent_reasoning:
            recent_reasoning = question

        guide_after = [g for g in guides if g.start() >= res_after[0].end()]
        qwq_sufficient, qwq_confidence = None, None
        if guide_after:
            g = guide_after[0]
            qwq_sufficient = (g.group(1) == "Sufficient")
            qwq_confidence = int(g.group(2)) / 100.0 if g.group(2) else (1.0 if qwq_sufficient else None)

        states.append({
            "dataset": dataset,
            "item_id": item.get("id"),
            "question": question,
            "golden_answers": golden_answers,
            "recent_reasoning": recent_reasoning,
            "search_query": q_match.group(1).strip(),
            "extracted_info": result_text,
            "qwq_sufficient": qwq_sufficient,
            "qwq_confidence": qwq_confidence,
        })
    return states


def main():
    all_states = []
    for dataset, batches in SRC_BATCHES.items():
        for dir_path, qa_path in batches:
            raw_path = latest_result_json(dir_path)
            if raw_path is None:
                print(f"[harvest] WARNING: no result json found in {dir_path}, skipping")
                continue
            question_lookup = load_question_lookup(qa_path)
            data = json.load(open(raw_path, encoding="utf-8"))
            n_before = len(all_states)
            for item in data:
                all_states.extend(extract_states(item, dataset, question_lookup))
            print(f"[harvest] {dataset} ({dir_path}): {len(data)} questions -> "
                  f"{len(all_states) - n_before} states (from {raw_path})")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in all_states:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[harvest] total {len(all_states)} states -> {OUT_PATH}")


if __name__ == "__main__":
    main()
