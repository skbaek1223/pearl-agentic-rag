import json
import os
import random
import re
import string
from itertools import groupby

IN_PATH = "outputs/evaluator_train/harvested_states.jsonl"
OUT_PATH = "outputs/evaluator_train/labeled_states.jsonl"
HOTPOTQA_TRAIN = "../data/eval/hotpotqa_train.jsonl"
WORD_OVERLAP_THRESHOLD = 0.7
MAX_CONSECUTIVE_INSUFFICIENT = 2


def normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def word_overlap(a, b):
    wa = set(normalize(a).split())
    wb = set(normalize(b).split())
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)


def load_hotpotqa_gold_sentences():
    lookup = {}
    with open(HOTPOTQA_TRAIN, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            meta = r.get("metadata", {})
            sf = meta.get("supporting_facts", {})
            ctx = meta.get("context", {})
            titles, sent_ids = sf.get("title", []), sf.get("sent_id", [])
            ctx_titles, ctx_sents = ctx.get("title", []), ctx.get("sentences", [])
            gold_sents = []
            for title, sid in zip(titles, sent_ids):
                if title in ctx_titles:
                    ti = ctx_titles.index(title)
                    sents = ctx_sents[ti] if ti < len(ctx_sents) else []
                    if 0 <= sid < len(sents):
                        gold_sents.append(sents[sid])
            lookup[r["question"].strip()] = gold_sents
    return lookup


def hotpotqa_gold_present(gold_sentences, extracted_info):
    return any(word_overlap(s, extracted_info) >= WORD_OVERLAP_THRESHOLD for s in gold_sentences)


def nq_gold_present(golden_answers, extracted_info):
    norm_info = normalize(extracted_info)
    return any(normalize(a) in norm_info for a in golden_answers if a)


def subsample_consecutive_insufficient(labeled_states, max_run=MAX_CONSECUTIVE_INSUFFICIENT, seed=0):
    rng = random.Random(seed)
    result = []
    for _, group_iter in groupby(labeled_states, key=lambda s: (s["dataset"], s["item_id"])):
        group = list(group_iter)
        i = 0
        while i < len(group):
            if group[i]["sufficient_label"] == 1:
                result.append(group[i])
                i += 1
                continue
            j = i
            while j < len(group) and group[j]["sufficient_label"] == 0:
                j += 1
            run = group[i:j]
            if len(run) <= max_run:
                result.extend(run)
            else:
                keep_idx = sorted(rng.sample(range(len(run)), max_run))
                result.extend(run[k] for k in keep_idx)
            i = j
    return result


def main():
    states = [json.loads(line) for line in open(IN_PATH, encoding="utf-8") if line.strip()]
    print(f"[build-labels] loaded {len(states)} states from {IN_PATH}")

    hotpotqa_gold = load_hotpotqa_gold_sentences()
    print(f"[build-labels] loaded gold supporting facts for {len(hotpotqa_gold)} hotpotqa questions")

    labeled = []
    n_gold, n_fallback, n_no_qwq_verdict = 0, 0, 0
    for s in states:
        if s["dataset"] == "hotpotqa":
            gold_sents = hotpotqa_gold.get(s["question"].strip(), [])
            gold_present = hotpotqa_gold_present(gold_sents, s["extracted_info"]) if gold_sents else False
        else:
            gold_present = nq_gold_present(s["golden_answers"], s["extracted_info"])

        if gold_present:
            correct_fraction = 1.0
            label_source = "gold"
            n_gold += 1
        elif s.get("qwq_sufficient") is not None:
            correct_fraction = 1.0 if s["qwq_sufficient"] else 0.0
            label_source = "qwq_fallback"
            n_fallback += 1
        else:
            n_no_qwq_verdict += 1
            continue

        labeled.append({
            **s,
            "correct_fraction": correct_fraction,
            "sufficient_label": int(correct_fraction >= 0.5),
            "label_source": label_source,
        })

    n_before_subsample = len(labeled)
    labeled = subsample_consecutive_insufficient(labeled)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in labeled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[build-labels] gold-matched={n_gold}  qwq_fallback={n_fallback}  "
          f"skipped(no verdict)={n_no_qwq_verdict}")
    print(f"[build-labels] after consecutive-insufficient subsampling (max {MAX_CONSECUTIVE_INSUFFICIENT}/run): "
          f"{n_before_subsample} -> {len(labeled)} states")
    print(f"[build-labels] saved {len(labeled)} labeled states -> {OUT_PATH}")
    fracs = [r["correct_fraction"] for r in labeled]
    print(f"[build-labels] mean correct_fraction={sum(fracs)/len(fracs):.3f}  "
          f"frac sufficient(>=0.5)={sum(r['sufficient_label'] for r in labeled)/len(labeled):.3f}")


if __name__ == "__main__":
    main()
