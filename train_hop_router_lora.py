import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-8B"


@dataclass
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    response_template: Union[str, List[int]] = None
    ignore_index: int = -100
    mlm: bool = False

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.response_template, str):
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            self.response_token_ids = list(self.response_template)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        examples = [
            {"input_ids": ex["input_ids"]} if hasattr(ex, "keys") else ex
            for ex in examples
        ]
        batch = super().torch_call(examples)
        for i in range(len(batch["labels"])):
            seq = batch["input_ids"][i].tolist()
            original_labels = batch["labels"][i].clone()
            batch["labels"][i] = torch.full_like(batch["labels"][i], self.ignore_index)
            response_starts = []
            for idx in range(len(seq) - len(self.response_token_ids) + 1):
                if seq[idx: idx + len(self.response_token_ids)] == self.response_token_ids:
                    response_starts.append(idx + len(self.response_token_ids))
            if not response_starts:
                warnings.warn("No response template found in an example; all labels masked.")
                continue
            for j, start in enumerate(response_starts):
                end = (
                    response_starts[j + 1] - len(self.response_token_ids)
                    if j + 1 < len(response_starts)
                    else len(seq)
                )
                batch["labels"][i, start:end] = original_labels[start:end]
        return batch


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--eval-steps", type=int, default=20,
                     help="single/multi classification converges fast on this small a task -- check often so early stopping can fire soon after it plateaus")
    ap.add_argument("--early-stopping-patience", type=int, default=2)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_raw = load_jsonl(args.train)
    val_raw = load_jsonl(args.val)
    print(f"train: {len(train_raw)}  val: {len(val_raw)}")

    def preprocess(examples):
        return {
            "text": [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in examples["messages"]
            ]
        }

    remove_cols = ["messages", "id"]
    train_ds = Dataset.from_list(train_raw).map(preprocess, batched=True, remove_columns=remove_cols)
    val_ds = Dataset.from_list(val_raw).map(preprocess, batched=True, remove_columns=remove_cols)

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n", tokenizer=tokenizer,
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=4,
        max_length=512,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"saved -> {final_dir}")


if __name__ == "__main__":
    main()
