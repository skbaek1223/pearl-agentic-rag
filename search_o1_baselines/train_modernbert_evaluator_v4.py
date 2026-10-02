import json

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments,
)
from datasets import Dataset

MODEL_NAME = "answerdotai/ModernBERT-large"
IN_PATH = "outputs/evaluator_train/labeled_states_v3_gpt5mini_verified.jsonl"
OUT_DIR = "outputs/evaluator_train/modernbert_evaluator_v4b"
MAX_LENGTH = 1024
SEED = 0
VAL_FRAC = 0.1


def build_text(state, repeat_question=False):
    if repeat_question:
        return (
            f"Question: {state['question']}\n"
            f"Prior reasoning: {state['recent_reasoning']}\n"
            f"Search query: {state['search_query']}\n"
            f"Does the following retrieved information directly answer the question "
            f"\"{state['question']}\"?\n"
            f"Retrieved information: {state['extracted_info']}"
        )
    return (
        f"Question: {state['question']}\n"
        f"Prior reasoning: {state['recent_reasoning']}\n"
        f"Search query: {state['search_query']}\n"
        f"Retrieved information: {state['extracted_info']}"
    )


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    import argparse
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of TRAIN rows to keep (stratified by class), for data-scaling ablation")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])
    ap.add_argument("--repeat-question", action="store_true",
                     help="repeat the question right before Retrieved information, with an explicit specificity-check framing")
    ap.add_argument("--extra-train-file", type=str, default=None,
                     help="JSONL of additional TRAIN-only records (e.g. synthetic hard negatives) to append after subsampling")
    ap.add_argument("--weight-power", type=float, default=1.0,
                     help="exponent applied to inverse-frequency class weights; 1.0=full, 0.0=equal weighting")
    cli_args = ap.parse_args()

    random.seed(SEED)
    states = [json.loads(line) for line in open(IN_PATH, encoding="utf-8") if line.strip()]
    print(f"[train-v4] loaded {len(states)} labeled states")

    questions = sorted({s["question"] for s in states})
    random.shuffle(questions)
    n_val_q = max(1, int(len(questions) * VAL_FRAC))
    val_qs = set(questions[:n_val_q])

    train_rows = [s for s in states if s["question"] not in val_qs]
    val_rows = [s for s in states if s["question"] in val_qs]
    print(f"[train-v4] split by question: {len(train_rows)} train states / {len(val_rows)} val states")

    if cli_args.frac < 1.0:
        rng = random.Random(SEED)
        pos_rows = [s for s in train_rows if s["correct_fraction"] >= 0.5]
        neg_rows = [s for s in train_rows if s["correct_fraction"] < 0.5]
        rng.shuffle(pos_rows)
        rng.shuffle(neg_rows)
        pos_rows = pos_rows[:max(1, int(len(pos_rows) * cli_args.frac))]
        neg_rows = neg_rows[:max(1, int(len(neg_rows) * cli_args.frac))]
        train_rows = pos_rows + neg_rows
        rng.shuffle(train_rows)
        print(f"[train-v4] subsampled to frac={cli_args.frac}: {len(train_rows)} train states "
              f"(pos={len(pos_rows)} neg={len(neg_rows)})")

    if cli_args.extra_train_file:
        extra_rows = [json.loads(l) for l in open(cli_args.extra_train_file, encoding="utf-8") if l.strip()]
        train_rows = train_rows + extra_rows
        print(f"[train-v4] added {len(extra_rows)} extra train-only records from {cli_args.extra_train_file} "
              f"-> {len(train_rows)} total train states")

    n_pos = sum(1 for s in train_rows if s["correct_fraction"] >= 0.5)
    n_neg = len(train_rows) - n_pos
    total = n_pos + n_neg
    raw_weights = [total / (2 * n_neg), total / (2 * n_pos)]
    class_weights = torch.tensor([w ** cli_args.weight_power for w in raw_weights], dtype=torch.float32)
    print(f"[train-v4] train class balance: neg={n_neg} pos={n_pos} -> weights={class_weights.tolist()}")

    def to_hf(rows):
        return Dataset.from_dict({
            "text": [build_text(s, repeat_question=cli_args.repeat_question) for s in rows],
            "label": [int(round(s["correct_fraction"])) for s in rows],
        })

    train_ds, val_ds = to_hf(train_rows), to_hf(val_rows)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, classifier_pooling=cli_args.pooling)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == labels).mean())
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        neg_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        neg_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        neg_f1 = (2 * neg_precision * neg_recall / (neg_precision + neg_recall)
                  if (neg_precision + neg_recall) > 0 else 0.0)
        macro_f1 = (f1 + neg_f1) / 2
        return {"acc": acc, "precision": precision, "recall": recall, "f1": f1,
                "neg_precision": neg_precision, "neg_recall": neg_recall, "neg_f1": neg_f1,
                "macro_f1": macro_f1}

    args = TrainingArguments(
        output_dir=cli_args.out_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        group_by_length=True,
        num_train_epochs=3,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        bf16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(f"{cli_args.out_dir}/final")
    tokenizer.save_pretrained(f"{cli_args.out_dir}/final")
    metrics = trainer.evaluate()
    print("[train-v4] final eval:", metrics)


if __name__ == "__main__":
    main()
