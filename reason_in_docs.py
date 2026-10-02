from __future__ import annotations

import json


def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing retrieved passages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Retrieved Passages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Retrieved Passages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Retrieved Passages:**
- Carefully review the content of each retrieved passage.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the retrieved passages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the passages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the passages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**
{prev_reasoning}

- **Current Search Query:**
{search_query}

- **Retrieved Passages:**
{document}

Now you should analyze each passage and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


INFOGEN_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)


def format_passages(passages: list[dict]) -> str:
    out = ""
    for i, p in enumerate(passages):
        doc = {"id": i + 1, "title": p["title"], "text": p["text"]}
        out += f"**Passage {i + 1}:**\n{json.dumps(doc, ensure_ascii=False, indent=2)}\n"
    return out


def truncate_prev_reasoning(output: str, markers: tuple[str, ...]) -> str:
    all_reasoning_steps = output.replace('\n\n', '\n').split("\n")
    truncated = ""
    for i, step in enumerate(all_reasoning_steps):
        truncated += f"Step {i + 1}: {step}\n\n"
    prev_steps = truncated.split('\n\n')
    if len(prev_steps) > 5:
        kept = ""
        for i, step in enumerate(prev_steps):
            if i == 0 or i >= len(prev_steps) - 4 or any(m in step for m in markers):
                kept += step + '\n\n'
            elif kept[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                kept += '...\n\n'
        truncated = kept
    return truncated.strip('\n')


def run_batch(llm, tokenizer, jobs: list[tuple[str, str, list[dict]]], markers: tuple[str, ...],
              max_model_len: int) -> tuple[list[str], int, int]:
    import sys
    from vllm import SamplingParams
    from baseline_eval import SEARCH_O1_WIKI_SCRIPTS
    if SEARCH_O1_WIKI_SCRIPTS not in sys.path:
        sys.path.insert(0, SEARCH_O1_WIKI_SCRIPTS)
    from evaluate import extract_answer

    if not jobs:
        return [], 0, 0
    prompts, sps = [], []
    for reasoning, query, passages in jobs:
        prev = truncate_prev_reasoning(reasoning, markers)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": get_webpage_to_reasonchain_instruction(prev, query, format_passages(passages))}],
            add_generation_prompt=True, tokenize=False)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > max_model_len - 1024:
            ids = ids[:max_model_len - 1024]
            text = tokenizer.decode(ids)
        prompts.append(text)
        sps.append(SamplingParams(max_tokens=min(32768, max_model_len - len(ids)), **INFOGEN_SAMPLING))
    outs = llm.generate(prompts, sampling_params=sps)
    infos = [extract_answer(o.outputs[0].text, mode="infogen") for o in outs]
    return (infos, sum(len(o.prompt_token_ids) for o in outs),
            sum(len(o.outputs[0].token_ids) for o in outs))
