import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_hop_router_lora import MODE_CONFIG, parse_label, MODEL_ID, PLANEVAL

ADAPTER_DIR = f"{PLANEVAL}/checkpoints/hop_router_lora_h500_nq200_tqa200_popqa100_answerfirst/final"
MODE = "answer_first"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="comma-separated subset of the 6 datasets")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--adapter-dir", default=None, help="override the ADAPTER_DIR module constant")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    adapter_dir = args.adapter_dir or ADAPTER_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    system_prompt = MODE_CONFIG[MODE]["system"]
    max_new_tokens = 60

    results = {}
    all_wrong = {}
    for ds in datasets:
        items = json.load(open(f"{PLANEVAL}/cache/{ds}_dev500.json", encoding="utf-8"))
        wrong = []
        n_correct = 0
        batch_size = 32
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            questions = [it["Question"] for it in batch]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for q in questions
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for it, t in zip(batch, texts):
                pred = parse_label(t, MODE)
                gold = it["gold_hop_label"]
                if pred == gold:
                    n_correct += 1
                else:
                    wrong.append({"id": it["id"], "question": it["Question"], "gold": gold,
                                  "pred": pred, "generated": t.strip()})
            print(f"  {ds}: {min(i + batch_size, len(items))}/{len(items)} (wrong so far: {len(wrong)})")
        acc = n_correct / len(items)
        results[ds] = {"accuracy": acc, "n": len(items), "n_wrong": len(wrong)}
        all_wrong[ds] = wrong
        print(f"=== {ds} dev500: accuracy={acc:.4f} ({len(wrong)} wrong) ===")

    tag = f"_{args.out_tag}" if args.out_tag else ""
    json.dump(results, open(f"{PLANEVAL}/outputs/hop_router_dev500_eval{tag}.json", "w"), indent=2)
    json.dump(all_wrong, open(f"{PLANEVAL}/outputs/hop_router_dev500_wrong{tag}.json", "w"),
              ensure_ascii=False, indent=2)
    print(f"\nsaved -> outputs/hop_router_dev500_eval{tag}.json, outputs/hop_router_dev500_wrong{tag}.json")


if __name__ == "__main__":
    main()
