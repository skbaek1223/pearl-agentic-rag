from __future__ import annotations

import argparse
import json
import os
import random

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

IN_PATH = os.environ.get(
    "PEARL_EVALUATOR_LABELED_STATES",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "search_o1_baselines", "outputs", "evaluator_train",
                 "labeled_states_v3_gpt5mini_verified.jsonl"),
)
SEED = 0
VAL_FRAC = 0.1
MAX_LENGTH = 1024


def build_text(state: dict) -> str:
    return (
        f"Question: {state['question']}\n"
        f"Prior reasoning: {state['recent_reasoning']}\n"
        f"Search query: {state['search_query']}\n"
        f"Retrieved information: {state['extracted_info']}"
    )


def reconstruct_val_split() -> list[dict]:
    states = [json.loads(l) for l in open(IN_PATH, encoding="utf-8") if l.strip()]
    random.seed(SEED)
    questions = sorted({s["question"] for s in states})
    random.shuffle(questions)
    n_val_q = max(1, int(len(questions) * VAL_FRAC))
    val_qs = set(questions[:n_val_q])
    val_rows = [s for s in states if s["question"] in val_qs]
    return val_rows


@torch.no_grad()
def run_inference(checkpoint: str, device: str, val_rows: list[dict], batch_size: int = 32):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()

    labels, confs, preds = [], [], []
    for start in range(0, len(val_rows), batch_size):
        batch = val_rows[start:start + batch_size]
        texts = [build_text(s) for s in batch]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH,
                         return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        batch_preds = torch.argmax(logits, dim=-1)
        for i, s in enumerate(batch):
            labels.append(int(round(s["correct_fraction"])))
            preds.append(int(batch_preds[i].item()))
            confs.append(probs[i, batch_preds[i]].item())
    return labels, preds, confs


def expected_calibration_error_by_pred_label(labels, preds, confs, n_bins: int = 10):
    for target_pred, name in [(1, "predicted SUFFICIENT"), (0, "predicted INSUFFICIENT")]:
        idx = [i for i, p in enumerate(preds) if p == target_pred]
        sub_labels = [labels[i] for i in idx]
        sub_preds = [preds[i] for i in idx]
        sub_confs = [confs[i] for i in idx]
        print(f"\n=== {name} (n={len(idx)}) ===")
        if not idx:
            print("  (no examples)")
            continue
        ece, rows = expected_calibration_error(sub_labels, sub_preds, sub_confs, n_bins)
        print(f"  ECE: {ece:.4f}")
        print(f"  {'conf bin':>14s} {'n':>6s} {'avg conf':>10s} {'accuracy':>10s} {'gap':>8s}")
        print("  " + "-" * 54)
        for lo, hi, n, avg_conf, acc_bin in rows:
            if n == 0:
                print(f"  {lo:5.2f}-{hi:5.2f} {n:6d} {'--':>10s} {'--':>10s} {'--':>8s}")
            else:
                gap = acc_bin - avg_conf
                print(f"  {lo:5.2f}-{hi:5.2f} {n:6d} {avg_conf:10.4f} {acc_bin:10.4f} {gap:+8.4f}")


def expected_calibration_error(labels, preds, confs, n_bins: int = 10):
    correct = [int(p == y) for p, y in zip(preds, labels)]
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confs, correct):
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append((c, ok))

    n = len(confs)
    ece = 0.0
    rows = []
    for i, bucket in enumerate(bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not bucket:
            rows.append((lo, hi, 0, None, None))
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(ok for _, ok in bucket) / len(bucket)
        weight = len(bucket) / n
        ece += weight * abs(acc - avg_conf)
        rows.append((lo, hi, len(bucket), avg_conf, acc))
    return ece, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    val_rows = reconstruct_val_split()
    print(f"Reconstructed held-out val split: {len(val_rows)} states "
          f"(seed={SEED}, val_frac={VAL_FRAC}, source={IN_PATH})")

    labels, preds, confs = run_inference(args.checkpoint, args.device, val_rows)

    acc = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Val accuracy (sanity check, should roughly match training logs): {acc:.4f}")

    ece, rows = expected_calibration_error(labels, preds, confs, args.n_bins)
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}  "
          f"(0 = perfectly calibrated, reported confidence == empirical accuracy)")
    print(f"\n{'conf bin':>14s} {'n':>6s} {'avg conf':>10s} {'accuracy':>10s} {'gap':>8s}")
    print("-" * 54)
    for lo, hi, n, avg_conf, acc_bin in rows:
        if n == 0:
            print(f"{lo:5.2f}-{hi:5.2f} {n:6d} {'--':>10s} {'--':>10s} {'--':>8s}")
        else:
            gap = acc_bin - avg_conf
            print(f"{lo:5.2f}-{hi:5.2f} {n:6d} {avg_conf:10.4f} {acc_bin:10.4f} {gap:+8.4f}")

    overconf_mass = sum(n for _, _, n, avg_conf, acc_bin in rows
                         if n and avg_conf is not None and avg_conf > (acc_bin or 0))
    print(f"\n{overconf_mass}/{len(labels)} val examples fall in bins where the model is "
          f"OVER-confident (avg confidence > actual accuracy in that bin).")

    n_pos_label = sum(labels)
    print(f"\nVal label balance: sufficient={n_pos_label} ({n_pos_label/len(labels):.1%})  "
          f"insufficient={len(labels)-n_pos_label} ({1-n_pos_label/len(labels):.1%})")
    expected_calibration_error_by_pred_label(labels, preds, confs, args.n_bins)


if __name__ == "__main__":
    main()
