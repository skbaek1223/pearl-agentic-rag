from __future__ import annotations

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PEARL_HOME = os.environ.get("PEARL_HOME", os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CHECKPOINT = os.environ.get(
    "PEARL_EVALUATOR_CHECKPOINT",
    os.path.join(PEARL_HOME, "checkpoints", "evaluator", "final"),
)
SUFFICIENT_LABEL = 1
MAX_LENGTH = 1024
MAX_FORWARD_BATCH = 64
SUFFICIENT_THRESHOLD = float(os.environ.get("EVALUATOR_THRESHOLD", "0.60"))


def build_text(question: str, recent_reasoning: str, search_query: str, extracted_info: str) -> str:
    return (
        f"Question: {question}\n"
        f"Prior reasoning: {recent_reasoning}\n"
        f"Search query: {search_query}\n"
        f"Retrieved information: {extracted_info}"
    )


class ModernBertEvaluator:
    def __init__(self, checkpoint_dir: str = DEFAULT_CHECKPOINT, device: str = "cuda",
                 threshold: float = SUFFICIENT_THRESHOLD):
        self.device = device
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, states: list[dict]) -> list[tuple[bool, float]]:
        if not states:
            return []
        out: list[tuple[bool, float]] = []
        for start in range(0, len(states), MAX_FORWARD_BATCH):
            chunk = states[start:start + MAX_FORWARD_BATCH]
            texts = [build_text(**s) for s in chunk]
            enc = self.tokenizer(
                texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            p_suff = probs[:, SUFFICIENT_LABEL]
            for i in range(len(chunk)):
                is_suff = bool(p_suff[i].item() >= self.threshold)
                conf = p_suff[i].item() if is_suff else 1.0 - p_suff[i].item()
                out.append((is_suff, conf))
        return out

    def predict_one(self, question: str, recent_reasoning: str, search_query: str,
                     extracted_info: str) -> tuple[bool, float]:
        return self.predict_batch([{
            "question": question, "recent_reasoning": recent_reasoning,
            "search_query": search_query, "extracted_info": extracted_info,
        }])[0]


class RemoteModernBertEvaluator:

    def __init__(self, url: str, timeout: float = 1800.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def predict_batch(self, states: list[dict]) -> list[tuple[bool, float]]:
        if not states:
            return []
        import requests
        resp = requests.post(f"{self.url}/predict_batch", json={"states": states},
                              timeout=self.timeout)
        resp.raise_for_status()
        return [(bool(is_suff), float(conf)) for is_suff, conf in resp.json()["results"]]

    def predict_one(self, question: str, recent_reasoning: str, search_query: str,
                     extracted_info: str) -> tuple[bool, float]:
        return self.predict_batch([{
            "question": question, "recent_reasoning": recent_reasoning,
            "search_query": search_query, "extracted_info": extracted_info,
        }])[0]
