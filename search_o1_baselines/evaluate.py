import re
import json
import math
import numpy as np
from collections import Counter
import string
import os, time


def extract_answer(output, mode='gen'):
    extracted_text = ''
    if mode == 'codegen':
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()
    elif mode == 'infogen':
        pattern_info = "**Final Information**"
        pattern_step = "**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        tag_matches = re.findall(r'<answer>(.*?)</answer>', output, re.DOTALL | re.IGNORECASE)
        if tag_matches:
            return tag_matches[-1].strip()
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]
            if mode in ['choose', 'qa']:
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]
                extracted_text = extracted_text.strip("()")
    return extracted_text


def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text

def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.strip().split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def evaluate_predictions(output, labeled_answer, mode='gen'):
    """Acc: 1 if a normalized gold answer is contained in the normalized prediction."""
    final_metric = {"acc": 0}
    pred_answer = extract_answer(output, mode=mode)

    if mode == 'qa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            # Count containment only when at least one word token is shared,
            # so a gold string matching inside another word does not score.
            shared = Counter(normalized_pred_answer.split()) & Counter(normalized_ground_truth.split())
            if sum(shared.values()) > 0 and normalized_ground_truth in normalized_pred_answer:
                final_metric["acc"] = 1
                break
    else:
        normalized_pred_answer = normalize_answer(pred_answer)
        normalized_ground_truth = normalize_answer(labeled_answer)
        final_metric["acc"] = int(normalized_ground_truth in normalized_pred_answer)

    return final_metric, pred_answer


_SEARCH_RESULT_RE = re.compile(
    r'<\|begin_search_result\|>.*?end_search_\w+\|?>', re.DOTALL)
_REASONING_GUIDE_RE = re.compile(r'\[Reasoning Guide\]:[^\n]*\n?')


def _strip_non_model_segments(text):
    text = _SEARCH_RESULT_RE.sub('', text)
    text = _REASONING_GUIDE_RE.sub('', text)
    return text


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SEARCH_QUERY_RE = re.compile(r'<\|begin_search_query\|>.*?<\|end_search_query\|>', re.DOTALL)
# Evaluator guide messages exactly as injected by run_agentic_rag.py
# (eval_guide_text); only the embedded question varies. The Insufficient
# template ends with "{question}." at the end of its line, so the question is
# matched to the line end rather than to its first period.
_GUIDE_SUFFICIENT_RE = re.compile(
    r'\[Reasoning Guide\]: Sufficient\.\n'
    r'1\) If there is still more information to retrieve before fully answering the original question ".*?", '
    r'derive an intermediate answer based on the retrieved information, then continue reasoning toward the next retrieval\. '
    r'You may refer to the reasoning context above or the retrieval guide to inform your search strategy, '
    r'or feel free to take a different approach\.\n'
    r'2\) If you have sufficient information to fully answer the original question, '
    r'provide the final answer in the format \\boxed\{YOUR_ANSWER\}\.\n?', re.DOTALL)
_GUIDE_INSUFFICIENT_RE = re.compile(
    r'\[Reasoning Guide\]: Insufficient\.\n'
    r'Feel free to explore alternative paths, such as trying a different search query or taking other retrieval steps as needed, '
    r'to derive an answer to the question: [^\n]*\.\n?')


def _extract_reasoning_chain(text):
    """Agent-written text used for Rep5/Ent: drop retrieved evidence, injected
    evaluator guides, and search queries."""
    text = _SEARCH_RESULT_RE.sub('', text)
    text = _GUIDE_SUFFICIENT_RE.sub('', text)
    text = _GUIDE_INSUFFICIENT_RE.sub('', text)
    text = _SEARCH_QUERY_RE.sub('', text)
    return text


def _tokenize_words(text):
    return [w.lower() for w in _WORD_RE.findall(text)]


def ngram_repetition_rate(words, n=5):
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    total = len(grams)
    unique = len(set(grams))
    return (total - unique) / total


def normalized_lexical_entropy(words):
    N = len(words)
    if N <= 1:
        return 0.0
    counts = Counter(words)
    H = -sum((c / N) * math.log2(c / N) for c in counts.values())
    denom = math.log2(N)
    if denom == 0:
        return 0.0
    return H / denom


def compute_rumination(text, n=5):
    chain = _extract_reasoning_chain(text)
    words = _tokenize_words(chain)
    return {
        'rumination_5gram_rep': ngram_repetition_rate(words, n=n),
        'rumination_lex_entropy': normalized_lexical_entropy(words),
    }


def _aggregate_rumination(rum_list):
    if not rum_list:
        return None
    rep = [r['rumination_5gram_rep'] for r in rum_list]
    ent = [r['rumination_lex_entropy'] for r in rum_list]
    return {
        'rumination_5gram_rep': float(np.mean(rep)),
        'rumination_lex_entropy': float(np.mean(ent)),
    }


def _avg_output_tokens(output_list, tokenizer):
    if tokenizer is None or not output_list:
        return None
    total = 0
    for r in output_list:
        text = r if isinstance(r, str) else r.outputs[0].text
        text = _strip_non_model_segments(text)
        total += len(tokenizer(text, add_special_tokens=False)['input_ids'])
    return total / len(output_list)


QA_DATASETS = ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle', 'ambigqa']


def _gold_answers(item):
    labeled_answer = item.get("golden_answers", item.get("answer"))
    if isinstance(labeled_answer, str):
        labeled_answer = [labeled_answer]
    return labeled_answer


def _score_items(data, dataset_name):
    """Score each item in place (Pred_Answer, Metrics, Rumination); return mean Acc and rumination."""
    if dataset_name not in QA_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name} (open-domain QA only: {QA_DATASETS})")
    accs, rum_list = [], []
    for item in data:
        metric, pred_answer = evaluate_predictions(
            output=item['Output'], labeled_answer=_gold_answers(item), mode='qa')
        rum = compute_rumination(item['Output'])
        item['Pred_Answer'] = pred_answer
        item['Metrics'] = metric
        item['Rumination'] = rum
        accs.append(metric['acc'])
        rum_list.append(rum)
    return (float(np.mean(accs)) if accs else 0.0), _aggregate_rumination(rum_list)


def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, apply_backoff=False, tokenizer=None):
    for item, input_prompt, result in zip(filtered_data, input_list, output_list):
        item['Output'] = result if isinstance(result, str) else result.outputs[0].text
        item['Question'] = input_prompt

    acc, rum_overall = _score_items(filtered_data, dataset_name)
    overall_metrics = {
        'acc': acc,
        'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
    }
    avg_tokens = _avg_output_tokens(output_list, tokenizer)
    if avg_tokens is not None:
        overall_metrics['avg_output_tokens'] = f'{avg_tokens:.1f}'
    if rum_overall is not None:
        overall_metrics['rumination'] = rum_overall

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump({'overall': overall_metrics}, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-score a saved prediction file (open-domain QA).")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the model output JSON file.')
    parser.add_argument('--dataset_name', type=str, required=True, choices=QA_DATASETS)
    parser.add_argument('--output_metrics_path', type=str, help='Path to save the evaluation metrics.')
    args = parser.parse_args()

    output_metrics_path = args.output_metrics_path or args.output_path.replace('.json', '.metrics.json')

    with open(args.output_path, mode='r', encoding='utf-8') as file:
        data = json.load(file)

    previous = {}
    if os.path.exists(output_metrics_path):
        with open(output_metrics_path, mode='r', encoding='utf-8') as file:
            previous = json.load(file).get('overall', {})

    acc, rum_overall = _score_items(data, args.dataset_name)
    overall_metrics = {'acc': acc}
    for key in ('query_latency', 'avg_output_tokens'):
        if key in previous:
            overall_metrics[key] = previous[key]
    if rum_overall is not None:
        overall_metrics['rumination'] = rum_overall

    with open(args.output_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)
    with open(output_metrics_path, mode='w', encoding='utf-8') as json_file:
        json.dump({'overall': overall_metrics}, json_file, indent=4, ensure_ascii=False)
