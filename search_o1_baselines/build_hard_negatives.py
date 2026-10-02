import json
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

IN_PATH = "outputs/evaluator_train/labeled_states_v3_gpt5mini_verified.jsonl"
OUT_PATH = "outputs/evaluator_train/hard_negatives_synthetic_topsim.jsonl"
SEED = 0
VAL_FRAC = 0.1
N_HARD_NEG_PER_DATASET = 700


def main():
    random.seed(SEED)
    states = [json.loads(line) for line in open(IN_PATH, encoding="utf-8") if line.strip()]

    questions = sorted({s["question"] for s in states})
    random.shuffle(questions)
    n_val_q = max(1, int(len(questions) * VAL_FRAC))
    val_qs = set(questions[:n_val_q])
    train_states = [s for s in states if s["question"] not in val_qs]
    print(f"[hard-neg] train pool: {len(train_states)} states (val excluded)")

    out_records = []
    for ds in ["nq", "hotpotqa"]:
        ds_states = [s for s in train_states if s["dataset"] == ds]
        pos_states = [s for s in ds_states if s["correct_fraction"] >= 0.5]
        print(f"[hard-neg] {ds}: {len(ds_states)} train states, {len(pos_states)} positive candidates")

        seen_q = set()
        pool_questions, pool_info = [], []
        for s in ds_states:
            if s["question"] not in seen_q:
                seen_q.add(s["question"])
                pool_questions.append(s["question"])
                pool_info.append(s["extracted_info"])

        vec = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vec.fit_transform(pool_questions)
        q_to_idx = {q: i for i, q in enumerate(pool_questions)}

        candidates = []
        for s in pos_states:
            qi = q_to_idx[s["question"]]
            sims = cosine_similarity(tfidf[qi], tfidf).flatten()
            sims[qi] = -1
            best_j = int(sims.argmax())
            if sims[best_j] <= 0.05:
                continue
            swapped_info = pool_info[best_j]
            if swapped_info.strip() == s["extracted_info"].strip():
                continue
            candidates.append((sims[best_j], s, best_j))

        candidates.sort(key=lambda x: x[0], reverse=True)
        n_made = 0
        for sim, s, best_j in candidates[:N_HARD_NEG_PER_DATASET]:
            out_records.append({
                "dataset": s["dataset"],
                "item_id": s["item_id"] + "_hardneg",
                "question": s["question"],
                "recent_reasoning": s["recent_reasoning"],
                "search_query": s["search_query"],
                "extracted_info": pool_info[best_j],
                "correct_fraction": 0.0,
                "sufficient_label": 0,
                "label_source": "synthetic_hard_negative",
                "swapped_from_question": pool_questions[best_j],
                "similarity": float(sim),
            })
            n_made += 1
        print(f"[hard-neg] {ds}: {len(candidates)} candidates scored, kept top {n_made} "
              f"(similarity range {candidates[N_HARD_NEG_PER_DATASET-1][0] if n_made>=N_HARD_NEG_PER_DATASET else candidates[-1][0]:.3f}"
              f"-{candidates[0][0]:.3f})")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[hard-neg] total {len(out_records)} synthetic hard negatives -> {OUT_PATH}")

    for r in out_records[:3]:
        print("\n---")
        print("Q:", r["question"])
        print("swapped-in extracted_info (from):", r["swapped_from_question"])
        print("extracted_info:", r["extracted_info"][:200])
        print("similarity:", r["similarity"])


if __name__ == "__main__":
    main()
