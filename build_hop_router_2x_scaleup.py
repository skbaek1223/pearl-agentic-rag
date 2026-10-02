import json
import os
import random
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

PLANEVAL = os.environ.get("PEARL_HOME", os.path.dirname(os.path.abspath(__file__)))
REGUIDE = os.environ.get("PEARL_GOALS_DATA_DIR", os.path.join(PLANEVAL, "data", "goals"))
SFT_DIR = Path(PLANEVAL) / "sft"
MODEL_PATH = "Qwen/Qwen3-8B"
SEED = 42

GOAL_FILES = {
    "hotpotqa": f"{REGUIDE}/hotpotqa_goals.jsonl",
    "triviaqa": f"{REGUIDE}/triviaqa_goals_ruleFour_filtered.jsonl",
}
LABELS = {"hotpotqa": "multi", "triviaqa": "single"}
NEW_SLICE = {"hotpotqa": (1200, 1500), "triviaqa": (950, 1200)}

ROUTER_SYSTEM_ANSWER_FIRST = """You classify whether answering a question requires SINGLE-HOP or MULTI-HOP retrieval.
SINGLE-HOP: the answer can be found by looking up one fact directly.
MULTI-HOP: answering requires first identifying an intermediate entity or fact, then using it to look up the final answer.
First write exactly "Answer: single" or "Answer: multi", then a period and ONE short justification sentence (name the specific fact, or the specific intermediate entity/fact chain, involved -- do not use the words "single-hop" or "multi-hop" in the justification)."""

GEN_SYSTEM = """You justify a given single-hop/multi-hop label for a question in ONE short sentence.
SINGLE-HOP: the answer can be found by looking up one fact directly.
MULTI-HOP: answering requires first identifying an intermediate entity or fact, then using it to look up the final answer.
Given the question and its correct label, write ONE short sentence (max ~20 words) justifying it: for "single" name the one fact that directly answers it; for "multi" name the specific intermediate entity/fact needed before the final lookup.
Do NOT use the words "single-hop" or "multi-hop" anywhere in your sentence -- describe the reasoning itself instead.
Output only the sentence, no preamble, no quotes."""


def eval_ids(dataset):
    items = json.load(open(f"{PLANEVAL}/cache/{dataset}_first1000.json", encoding="utf-8"))
    return {it["id"] for it in items}


def load_goalfile_pool(dataset, cap, exclude_ids):
    items = [json.loads(l) for l in open(GOAL_FILES[dataset], encoding="utf-8")]
    random.Random(SEED).shuffle(items)
    qs = []
    seen = set()
    for it in items:
        if it.get("id") in exclude_ids:
            continue
        q = (it.get("question") or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        qs.append(q)
        if len(qs) >= cap:
            break
    return qs


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def idx(r):
    return int(r["id"].split("_")[-1].replace("dup", "").replace("rep0", "").replace("rep1", "").replace("rep2", ""))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(model=MODEL_PATH, gpu_memory_utilization=0.85, max_model_len=2048, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=60)

    def build_gen_prompt(question, label):
        msgs = [
            {"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": f"Question: {question}\nLabel: {label}"},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    new_records_by_ds = {}
    for ds, (start, end) in NEW_SLICE.items():
        pool = load_goalfile_pool(ds, end, eval_ids(ds))
        new_qs = pool[start:end]
        print(f"{ds}: pool={len(pool)}, new slice [{start}:{end}) -> {len(new_qs)} questions")
        prompts = [build_gen_prompt(q, LABELS[ds]) for q in new_qs]
        outs = llm.generate(prompts, sp)
        recs = []
        for i, (q, out) in enumerate(zip(new_qs, outs)):
            rationale = out.outputs[0].text.strip().split("\n")[0].strip().strip('"')
            assistant_content = f"Answer: {LABELS[ds]}. {rationale}"
            recs.append({
                "id": f"{ds}_scaleup2x_{start + i}",
                "messages": [
                    {"role": "system", "content": ROUTER_SYSTEM_ANSWER_FIRST},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": assistant_content},
                ],
            })
        new_records_by_ds[ds] = recs
        print(f"  sample: {recs[0]['messages'][1]['content'][:60]} -> {recs[0]['messages'][2]['content']}")

    h700_train = load_jsonl(SFT_DIR / "hop_router_h700_nq200_tqa400_popqa100_answerfirst_train.jsonl")
    h700_val = load_jsonl(SFT_DIR / "hop_router_h700_nq200_tqa400_popqa100_answerfirst_val.jsonl")
    h700_all = h700_train + h700_val

    h500nq250_train = load_jsonl(SFT_DIR / "hop_router_h500_nq250_tqa250_answerfirst_train.jsonl")
    h500nq250_val = load_jsonl(SFT_DIR / "hop_router_h500_nq250_tqa250_answerfirst_val.jsonl")
    h500nq250_all = h500nq250_train + h500nq250_val

    comp_train = load_jsonl(SFT_DIR / "hop_router_h500_nq100_tqa300comp_popqa100_answerfirst_train.jsonl")
    comp_val = load_jsonl(SFT_DIR / "hop_router_h500_nq100_tqa300comp_popqa100_answerfirst_val.jsonl")
    comp_all = comp_train + comp_val

    hotpotqa_orig = [r for r in h700_all if r["id"].startswith("hotpotqa_") and "scaleup" not in r["id"]]
    hotpotqa_cached_scaleup = [r for r in h700_all if r["id"].startswith("hotpotqa_scaleup_")]
    hotpotqa_new = new_records_by_ds["hotpotqa"]
    hotpotqa_all = hotpotqa_orig + hotpotqa_cached_scaleup + hotpotqa_new
    print(f"\nhotpotqa: {len(hotpotqa_orig)} orig + {len(hotpotqa_cached_scaleup)} cached-scaleup + "
          f"{len(hotpotqa_new)} new = {len(hotpotqa_all)}")
    assert len(hotpotqa_all) == 1000

    nq_all = [r for r in h500nq250_all if r["id"].startswith("nq_") and idx(r) < 200]
    print(f"nq: {len(nq_all)}")
    assert len(nq_all) == 200

    tqa_existing = [r for r in comp_all if r["id"].startswith("triviaqa_") and "comp" not in r["id"]
                    and "scaleup" not in r["id"]]
    tqa_scaleup_used_ids = {idx(r) for r in comp_all if r["id"].startswith("triviaqa_scaleup_")}
    print(f"triviaqa_scaleup ids already used in comp300 set: {sorted(tqa_scaleup_used_ids)[:3]}...")
    tqa_scaleup_all_cached = [r for r in h700_all if r["id"].startswith("triviaqa_scaleup_")]
    tqa_scaleup_unused = [r for r in tqa_scaleup_all_cached if idx(r) not in tqa_scaleup_used_ids]
    tqa_new = new_records_by_ds["triviaqa"]
    tqa_noncomp_all = tqa_existing + tqa_scaleup_unused + tqa_new
    print(f"triviaqa non-comp: {len(tqa_existing)} existing + {len(tqa_scaleup_unused)} cached-unused-scaleup + "
          f"{len(tqa_new)} new = {len(tqa_noncomp_all)}")
    assert len(tqa_noncomp_all) == 600, len(tqa_noncomp_all)

    comp37 = load_jsonl(SFT_DIR / "hop_router_triviaqa_comparison37_answerfirst.jsonl")
    random.Random(SEED).shuffle(comp37)
    comp100 = []
    for i in range(100):
        src = comp37[i % len(comp37)]
        rec = dict(src, id=f"{src['id']}_x{i // len(comp37)}")
        comp100.append(rec)
    print(f"comp: {len(comp37)} unique oversampled to {len(comp100)}")

    popqa_all = [r for r in comp_all if r["id"].startswith("popqa_")]
    print(f"popqa: {len(popqa_all)} (unchanged)")
    assert len(popqa_all) == 100

    all_records = hotpotqa_all + nq_all + tqa_noncomp_all + comp100 + popqa_all
    n_multi = len(hotpotqa_all)
    n_single = len(all_records) - n_multi
    print(f"\nGRAND total: {len(all_records)} (multi={n_multi}, single={n_single})")
    assert n_multi == n_single == 1000

    random.Random(SEED).shuffle(all_records)
    n_val = int(len(all_records) * 0.1)
    val_records = all_records[:n_val]
    train_records = all_records[n_val:]

    def dump(records, name):
        path = SFT_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(records)})")

    dump(train_records, "hop_router_h1000_nq200_tqa700comp_popqa100_answerfirst_train.jsonl")
    dump(val_records, "hop_router_h1000_nq200_tqa700comp_popqa100_answerfirst_val.jsonl")


if __name__ == "__main__":
    main()
