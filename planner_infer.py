from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PEARL_HOME = os.environ.get("PEARL_HOME", os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ADAPTER_DIR = os.environ.get(
    "PEARL_PLANNER_ADAPTER_DIR",
    os.path.join(PEARL_HOME, "checkpoints", "planner_lora", "final"),
)
BASE_MODEL_ID = "Qwen/Qwen3-8B"

MAX_PROMPT_TOKENS = 4096

SYSTEM_PROMPT = (
    "You are an information retrieval planning expert. Given a question, "
    "generate a single retrieval step that finds the answer directly.\n\n"
    "Write exactly one numbered step. No other text."
)

MULTIHOP_SYSTEM_PROMPT = (
    "You are an information retrieval planning expert. Given a question, "
    "generate an ordered sequence of concrete retrieval steps required to "
    "find the answer. Each step represents one retrieval action.\n\n"
    "Write one step per line, numbered. No other text."
)

FEWSHOT: list[dict[str, str]] = [
    {
        "q": "when was the first home pregnancy test invented",
        "plan": "1. Find when the first home pregnancy test was invented",
    },
    {
        "q": "what state of water is most common on earth's surface",
        "plan": "1. Identify the most common state of water found on Earth's surface",
    },
]

MULTIHOP_FEWSHOT: list[dict[str, str]] = [
    {
        "q": "What actor that was born in January 22, 1975 appeared in the film Out in fifty?",
        "plan": (
            "1. List the cast of the film 'Out in Fifty' to identify actors who appeared in it\n"
            "2. From the cast list, identify which actor was born on January 22, 1975"
        ),
    },
    {
        "q": "Which country the director of film Big Nothing is from?",
        "plan": (
            "1. Identify the director of the film 'Big Nothing'\n"
            "2. Find the director's nationality"
        ),
    },
]

MULTIHOP_MIXED_FEWSHOT: list[dict[str, str]] = [
    MULTIHOP_FEWSHOT[0],
    {
        "q": "In which city were the Winter Olympic Games held when John Curry won a gold medal for men's figure skating?",
        "plan": (
            "1. Find the year John Curry won the Olympic gold medal in men's figure skating\n"
            "2. Find the host city of the Winter Olympic Games held in that year"
        ),
    },
    MULTIHOP_FEWSHOT[1],
]


HOP_ROUTER_LORA_DIR = os.environ.get(
    "PEARL_HOP_ROUTER_ADAPTER_DIR",
    os.path.join(PEARL_HOME, "checkpoints", "hop_router_lora_scale2", "final"),
)
HOP_ROUTER_THRESHOLD = 0.9
ROUTER_SYSTEM_ANSWER_FIRST = """You classify whether answering a question requires SINGLE-HOP or MULTI-HOP retrieval.
SINGLE-HOP: the answer can be found by looking up one fact directly.
MULTI-HOP: answering requires first identifying an intermediate entity or fact, then using it to look up the final answer.
First write exactly "Answer: single" or "Answer: multi", then a period and ONE short justification sentence (name the specific fact, or the specific intermediate entity/fact chain, involved -- do not use the words "single-hop" or "multi-hop" in the justification)."""

STEP1_TYPE_SYSTEM_PROMPT = """You classify a planning step (with its leading verb removed, e.g. "the energy range of X" rather than "Search for the energy range of X") as LOOKUP or REASON, given the full question it comes from for context.
LOOKUP: the step names a specific fact, constant, named phenomenon, or source that must be retrieved from an external reference -- it cannot be derived from general domain knowledge or the values already given in the question.
REASON: the step names a calculation, standard reaction, or well-known relationship that can be worked out directly from values/entities already given in the question, using knowledge a domain expert would already have, without needing to look anything up.
First write exactly "Answer: lookup" or "Answer: reason", then a period and ONE short justification sentence."""

STEP1_TYPE_FEWSHOT: list[dict[str, str]] = [
    {
        "q": "What is the energy range of pp III neutrinos?",
        "step": "the energy range of pp III neutrinos",
        "answer": "lookup. This names a specific empirical fact about neutrino energy spectra that must be retrieved from a physics reference.",
    },
    {
        "q": (
            "1-bromopropane is treated with magnesium in anhydrous diethyl "
            "ether to form product 1. Product 1 is then reacted with "
            "acetaldehyde, followed by acidic aqueous workup, to form "
            "product 2. What is product 2?"
        ),
        "step": "the product formed when product 1 (propylmagnesium bromide) reacts with acetaldehyde followed by acidic aqueous workup",
        "answer": "reason. This is a standard Grignard addition to an aldehyde, a well-known reaction a chemist already knows how to work out.",
    },
    {
        "q": "Using the electron's rest mass energy, find the threshold energy for electron-positron pair production.",
        "step": "the rest mass energy of the electron",
        "answer": "lookup. This is a specific physical constant that must be retrieved, not derived.",
    },
    {
        "q": (
            "Light of wavelength 550 nm passes through a circular aperture "
            "of diameter 0.2 mm. What is the angular distance between the "
            "first two diffraction minima?"
        ),
        "step": "the angular distance between the first two diffraction minima using the circular-aperture formula with the given wavelength and aperture diameter",
        "answer": "reason. This is a direct application of a known formula to values already given in the question.",
    },
]

_STEP_RE = re.compile(r"^\s*\d+[\.\)]\s*(.+)", re.MULTILINE)


def _parse_steps(text: str) -> list[str]:
    steps = [m.group(1).strip() for m in _STEP_RE.finditer(text)]
    if not steps:
        steps = [text.strip()] if text.strip() else []
    return steps


class Planner:
    def __init__(self, adapter_dir: str | None = DEFAULT_ADAPTER_DIR,
                 base_model_id: str = BASE_MODEL_ID, device_map: str = "auto"):
        self._adapter_dir = adapter_dir
        self._base_model_id = base_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, torch_dtype=torch.bfloat16, device_map=device_map,
            trust_remote_code=True,
        )
        self._has_adapter = not (adapter_dir is None or str(adapter_dir).lower() == "none")
        if not self._has_adapter:
            self.model = base_model
        else:
            self.model = PeftModel.from_pretrained(base_model, adapter_dir, adapter_name="default")
            self.model.load_adapter(HOP_ROUTER_LORA_DIR, adapter_name="hop_router")
            self.model.set_adapter("default")
        self.model.eval()
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    @torch.no_grad()
    def generate_plans(self, questions: list[str], max_new_tokens: int = 256,
                        batch_size: int = 8, is_multi_hop: bool = False,
                        dataset_name: str | None = None) -> list[list[str]]:
        all_steps: list[list[str]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        pad_id = self.tokenizer.pad_token_id
        if is_multi_hop:
            fewshot = MULTIHOP_MIXED_FEWSHOT if self._has_adapter else MULTIHOP_FEWSHOT
        else:
            fewshot = FEWSHOT
        system_prompt = MULTIHOP_SYSTEM_PROMPT if is_multi_hop else SYSTEM_PROMPT
        fewshot_msgs = [
            m
            for ex in fewshot
            for m in (
                {"role": "user", "content": f"Question: {ex['q']}"},
                {"role": "assistant", "content": ex["plan"]},
            )
        ]
        for start in range(0, len(questions), batch_size):
            batch = questions[start:start + batch_size]
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}]
                    + fewshot_msgs
                    + [{"role": "user", "content": f"Question: {q}"}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for q in batch
            ]
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_PROMPT_TOKENS
            ).to(self.model.device)
            total_prompt_tokens += int(inputs["attention_mask"].sum().item())
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            gen_only = outputs[:, inputs["input_ids"].shape[1]:]
            total_completion_tokens += int((gen_only != pad_id).sum().item())
            texts = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            all_steps.extend(_parse_steps(t) for t in texts)
            del inputs, outputs, gen_only
            torch.cuda.empty_cache()
        self.last_prompt_tokens = total_prompt_tokens
        self.last_completion_tokens = total_completion_tokens
        return all_steps

    @torch.no_grad()
    def _classify_hop(self, questions: list[str], batch_size: int = 8,
                       threshold: float = HOP_ROUTER_THRESHOLD) -> list[str]:
        single_id = self.tokenizer.encode(" single", add_special_tokens=False)[0]
        multi_id = self.tokenizer.encode(" multi", add_special_tokens=False)[0]
        labels: list[str] = []
        self.model.set_adapter("hop_router")
        for start in range(0, len(questions), batch_size):
            batch = questions[start:start + batch_size]
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": ROUTER_SYSTEM_ANSWER_FIRST},
                     {"role": "user", "content": q}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                ) + "Answer:"
                for q in batch
            ]
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_PROMPT_TOKENS
            ).to(self.model.device)
            logits = self.model(**inputs, logits_to_keep=1).logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            p_single = probs[:, single_id]
            p_multi = probs[:, multi_id]
            p_multi_norm = p_multi / (p_single + p_multi + 1e-9)
            labels.extend("multi" if p.item() > threshold else "single" for p in p_multi_norm)
            del inputs, logits, probs
            torch.cuda.empty_cache()
        self.model.set_adapter("default")
        return labels

    @torch.no_grad()
    def classify_step1_type(self, questions: list[str], stripped_step1_texts: list[str],
                             batch_size: int = 8) -> list[str]:
        lookup_id = self.tokenizer.encode(" lookup", add_special_tokens=False)[0]
        reason_id = self.tokenizer.encode(" reason", add_special_tokens=False)[0]
        fewshot_msgs = [
            m
            for ex in STEP1_TYPE_FEWSHOT
            for m in (
                {"role": "user", "content": f"Question: {ex['q']}\nStep: {ex['step']}"},
                {"role": "assistant", "content": f"Answer: {ex['answer']}"},
            )
        ]
        labels: list[str] = []

        def _run(model):
            for start in range(0, len(questions), batch_size):
                q_batch = questions[start:start + batch_size]
                s_batch = stripped_step1_texts[start:start + batch_size]
                prompts = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "system", "content": STEP1_TYPE_SYSTEM_PROMPT}]
                        + fewshot_msgs
                        + [{"role": "user", "content": f"Question: {q}\nStep: {s}"}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    ) + "Answer:"
                    for q, s in zip(q_batch, s_batch)
                ]
                inputs = self.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_PROMPT_TOKENS
                ).to(self.model.device)
                logits = model(**inputs, logits_to_keep=1).logits[:, -1, :]
                probs = torch.softmax(logits.float(), dim=-1)
                p_lookup = probs[:, lookup_id]
                p_reason = probs[:, reason_id]
                p_reason_norm = p_reason / (p_lookup + p_reason + 1e-9)
                labels.extend("reason" if p.item() > 0.5 else "lookup" for p in p_reason_norm)
                del inputs, logits, probs
                torch.cuda.empty_cache()

        if self._has_adapter:
            with self.model.disable_adapter():
                _run(self.model)
        else:
            _run(self.model)
        return labels

    @torch.no_grad()
    def route_and_generate_plans(self, questions: list[str], max_new_tokens: int = 256,
                                  batch_size: int = 8,
                                  dataset_name: str | None = None,
                                  skip_planner_for_single: bool = False) -> tuple[list[list[str]], list[str]]:
        if not self._has_adapter:
            raise ValueError(
                "route_and_generate_plans requires a real adapter_dir (trained "
                "planner) -- there is nothing to route TO otherwise. Use plain "
                "generate_plans() for an all-untrained or all-trained run."
            )
        num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(","))
        worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "planner_vllm_worker.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "questions.json")
            out_path = os.path.join(tmpdir, "result.json")
            with open(in_path, "w", encoding="utf-8") as f:
                json.dump({"questions": questions, "dataset_name": dataset_name}, f)
            subprocess.run(
                [
                    sys.executable, worker_path,
                    "--questions_file", in_path,
                    "--out_file", out_path,
                    "--base_model", self._base_model_id,
                    "--adapter_dir", self._adapter_dir,
                    "--hop_router_dir", HOP_ROUTER_LORA_DIR,
                    "--max_new_tokens", str(max_new_tokens),
                    "--threshold", str(HOP_ROUTER_THRESHOLD),
                    "--tensor_parallel_size", str(num_gpus),
                    "--gpu_memory_utilization", "0.45",
                ]
                + (["--skip_single"] if skip_planner_for_single else []),
                check=True,
            )
            with open(out_path, encoding="utf-8") as f:
                result = json.load(f)

        plans: list[list[str] | None] = result["plans"]
        hop_labels: list[str] = result["hop_labels"]
        self.last_prompt_tokens = result["prompt_tokens"]
        self.last_completion_tokens = result["completion_tokens"]
        assert all(p is not None for p in plans)
        return plans, hop_labels
