# PEARL: Planner and Evaluator for Reasoning-Efficient Agentic RAG

This repository contains the reproducibility code for **PEARL**, an inference-time
framework that guides a large reasoning model (LRM) through an agentic
retrieval-augmented generation (RAG) loop using hop-aware retrieval planning
and an evidence-sufficiency evaluator, plus the baseline and evaluation code
needed to reproduce the paper's experiments.

This is a code-availability release for double-blind review. Trained
checkpoints (planner LoRA, hop-router LoRA, ModernBERT evaluator) and the
retrieval corpus/index are **not included** due to size; training/build
scripts for each are provided below.

## Directory layout

Core PEARL pipeline (repo root):
- `run_agentic_rag.py` — main agentic-RAG loop, including all ablation flags
  (`--use_hop_router`, `--no_planner`, `--no_evaluator`,
  `--forced_search_question_only`, etc.)
- `run_agentic_rag_fsq_noeval.py` — narrow fork used to regenerate part of the
  "w/o Evaluator" ablation for single-routed questions
- `planner_infer.py` — planner + hop-router inference (Qwen3-8B + LoRA).
  `route_and_generate_plans()` (the `--use_hop_router` path) shells out to
  `planner_vllm_worker.py` for the actual hop-classification + plan-generation
  step (vLLM continuous batching instead of an in-process HF/PEFT loop, which
  measured 1h15m+ per 1000-question dataset under naive multi-GPU pipeline
  parallelism); the plain `generate_plans()`/`classify_step1_type()` methods
  still run in-process.
- `planner_vllm_worker.py` — standalone vLLM subprocess used by
  `planner_infer.py`'s `route_and_generate_plans()`; must stay in the same
  directory as `planner_infer.py` (invoked via
  `subprocess.run([sys.executable, <dir>/planner_vllm_worker.py, ...])`)
- `precompute_plans.py` — vLLM-based offline plan precomputation/caching
  using the trained planner adapter (faster than `planner_infer.py`'s
  in-process HF loop for large datasets)
- `modernbert_evaluator.py` — evaluator inference wrapper (in-process or via
  `evaluator_server.py`)
- `evaluator_server.py` — FastAPI server exposing the evaluator over HTTP
- `local_prompts.py` / `local_prompts_fsq_noeval.py` — prompt templates
- `train_hop_router_lora.py` — hop-router LoRA training
- `build_hop_router_2x_scaleup.py` — builds the final hop-router SFT dataset
  composition reported in the paper
- `eval_hop_router_dev.py` — dev-set evaluation for hop-router threshold
  selection
- `merge_final_results.py` — merges per-dataset/per-shard result files
- `run_webdancer.py` — WebDancer-32B baseline using ASearcher's local
  Wikipedia page corpus + E5 re-ranking for its `visit` tool, as described in
  the paper's Appendix "Baseline Adaptation". **This is the script that
  produced the paper's Table 1 WebDancer numbers.** Depends on
  `webdancer_prompts.py`, `baseline_eval.py`, `reason_in_docs.py`,
  `page_chunks.py`, `wiki_browse_backend.py`, and (offline, to build the page
  index consumed by `wiki_browse_backend.py`) `build_asearcher_page_index.py`.
- `run_webdancer_wiki.py` — a simpler WebDancer-32B port whose `visit` tool
  serves pages from an in-memory cache of already-searched passages (no
  separate page corpus). **Used only for the Appendix controlled-latency
  table, not Table 1** — do not substitute it for `run_webdancer.py` when
  reproducing the main results. Depends on `webdancer_prompts.py`.
- `check_evaluator_calibration.py` — evaluator confidence calibration check
- `sft/` — final hop-router SFT training/validation data (JSONL; small,
  included directly)

`search_o1_baselines/` (vendored, adapted from the Search-o1 codebase this
project was forked from):
- `run_search_o1_wiki.py`, `run_search_o1_wiki_nowait.py`,
  `run_search_o1_wiki_deer.py` — Search-o1 / NoWait / DEER baselines
- `evaluate.py` — EM/Acc/F1 scoring and the 5-gram repetition / normalized
  lexical entropy ("rumination") metrics reported in the paper
- `retriever_server.py`, `retriever_utils.py` — dense retriever (E5-base-v2
  over Wikipedia-2018) serving code
- `prompts.py` — base Search-o1-family prompts
- `train_modernbert_evaluator_v4.py`, `build_hard_negatives.py`,
  `build_final_labels.py`, `harvest_evaluator_states.py` — the evaluator
  training data pipeline (code only; no trained weights)

## Environment variables

No absolute paths are hardcoded. Configure via:
- `PEARL_HOME` — repo root (defaults to the directory containing the script)
- `PEARL_QWQ_MODEL_PATH` — path to the QwQ-32B checkpoint used as the
  reasoning LRM (also used for extraction calls)
- `PEARL_PLANNER_ADAPTER_DIR` — trained planner LoRA adapter (defaults to
  `$PEARL_HOME/checkpoints/planner_lora/final`)
- `PEARL_HOP_ROUTER_ADAPTER_DIR` — trained hop-router LoRA adapter (defaults
  to `$PEARL_HOME/checkpoints/hop_router_lora_scale2/final`)
- `PEARL_EVALUATOR_CHECKPOINT` — trained ModernBERT evaluator checkpoint
  (defaults to `$PEARL_HOME/checkpoints/evaluator/final`)
- `PEARL_E5_INDEX_PATH` / `PEARL_WIKI_CORPUS_PATH` — FAISS index and
  Wikipedia-2018 passage corpus for `search_o1_baselines/retriever_server.py`
- `PEARL_GOALS_DATA_DIR` — source pool of question/plan "goal" files consumed
  by `build_hop_router_2x_scaleup.py`
- `PEARL_WEBDANCER_MODEL_PATH` — path to the WebDancer-32B checkpoint, used
  by both `run_webdancer.py` and `run_webdancer_wiki.py` (`--model_path`)
- `PEARL_E5_MODEL_PATH` — path to the e5-base-v2 checkpoint used by
  `page_chunks.py` for page-chunk selection in `run_webdancer.py`
  (`--e5_model_path`)
- `PEARL_ASEARCHER_PAGES_PATH` — path to ASearcher's local-knowledge
  `wiki_webpages.jsonl` (huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge),
  used by `build_asearcher_page_index.py` (`--pages`) and
  `wiki_browse_backend.py`
- `--retriever_url` (CLI flag, all run scripts) — URL of a running
  `retriever_server.py` instance, e.g. `http://127.0.0.1:8765`
- `--evaluator_url` (CLI flag, optional) — URL of a running
  `evaluator_server.py` instance, to avoid loading the evaluator in-process
  on the same GPU as vLLM
- `EVALUATOR_THRESHOLD` — overrides the evaluator's sufficiency threshold
  (float in `[0,1]`); the deployed checkpoint's own default is `0.60`. Note:
  the paper's primary experiments use `\tau_e = 0.5` and report `0.6` only as
  a sensitivity check (Appendix "Evaluator Threshold") — pass
  `EVALUATOR_THRESHOLD=0.5` (or construct `ModernBertEvaluator`/
  `RemoteModernBertEvaluator` with `threshold=0.5`) to match the main results.
  `evaluator_server.py` also accepts a `--threshold` CLI override (defaults to
  this same value) and reports the active threshold in its response.

API keys (OpenAI, etc., where applicable to data-prep scripts) are read via
`os.environ`/`os.getenv` and are never hardcoded.

## What is not reproduced here

Per the paper (Appendix "Prompts"), the offline GPT-5 prompt used to generate
the planner's teacher training plans is a separate component and is not
reproduced in this repository. `precompute_plans.py` and `planner_infer.py`
only *consume* an already-trained planner adapter; they make no GPT-5/OpenAI
calls.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

You will additionally need:
1. A Wikipedia-2018 passage corpus + E5-base-v2 FAISS index (e.g. via
   [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)), served by
   `search_o1_baselines/retriever_server.py`.
2. A local QwQ-32B checkpoint for the reasoning LRM.
3. Trained planner LoRA, hop-router LoRA, and ModernBERT evaluator
   checkpoints — build these with `train_hop_router_lora.py` and the
   `search_o1_baselines/` evaluator-training pipeline (planner training code
   itself follows the same QLoRA recipe described in the paper's Appendix
   "Planner Training"; the teacher-plan generation step is external, see
   above).
4. For the WebDancer baseline (`run_webdancer.py`/`run_webdancer_wiki.py`): a
   local WebDancer-32B checkpoint and an e5-base-v2 checkpoint
   (`PEARL_WEBDANCER_MODEL_PATH`/`PEARL_E5_MODEL_PATH`). `run_webdancer.py`
   additionally needs ASearcher's local-knowledge `wiki_webpages.jsonl`
   (huggingface.co/datasets/inclusionAI/ASearcher-Local-Knowledge,
   `PEARL_ASEARCHER_PAGES_PATH`) and its byte-offset index, built once via
   `python build_asearcher_page_index.py --out_dir cache/asearcher_page_index`.

## Running

Start the retriever and (optionally) the evaluator server, then run e.g.:

```bash
python search_o1_baselines/retriever_server.py --gpus 0,1 --port 8765
python run_agentic_rag.py --dataset_name hotpotqa \
    --qa_data_path /path/to/hotpotqa_test.json \
    --retriever_url http://127.0.0.1:8765 \
    --use_hop_router --subset_num 1000
```

See each script's `--help` / module docstring for the full set of ablation
and configuration flags.
