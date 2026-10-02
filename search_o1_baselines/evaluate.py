import re
import json
import math
import numpy as np
from collections import Counter
import string
import os, time
from collections import defaultdict
try:
    from lcb_runner.evaluation import codegen_metrics
except Exception:
    codegen_metrics = None
try:
    from utils.math_equivalence import is_equiv
except Exception:
    def is_equiv(a, b):
        return 0


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
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}
    pred_answer = extract_answer(output, mode=mode)
    if pred_answer != '':
        final_metric["is_valid_answer"] = True

    if mode == 'qa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    else:
        normalized_pred_answer = normalize_answer(pred_answer)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)
    
        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

    return final_metric, pred_answer


_SEARCH_RESULT_RE = re.compile(
    r'<\|begin_search_result\|>.*?end_search_\w+\|?>', re.DOTALL)
_REASONING_GUIDE_RE = re.compile(r'\[Reasoning Guide\]:[^\n]*\n?')


def _strip_non_model_segments(text):
    text = _SEARCH_RESULT_RE.sub('', text)
    text = _REASONING_GUIDE_RE.sub('', text)
    return text


_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _extract_reasoning_chain(text):
    text = _strip_non_model_segments(text)
    matches = _THINK_RE.findall(text)
    if matches:
        return "\n".join(matches)
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


def _extract_judge_response_text(output: str) -> str:
    return output[-2000:]


def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, apply_backoff=False, tokenizer=None):
    avg_tokens = _avg_output_tokens(output_list, tokenizer)
    if dataset_name == 'livecode':
        samples_list = []
        generations_list = []

        difficulties = []
        per_difficulty_count = {}
        num_valid_answer = 0

        rum_list = []
        rum_per_difficulty = defaultdict(list)
        for item, input_prompt, result in zip(filtered_data, input_list, output_list):
            if type(result) == str:
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text
            difficulty = item.get("difficulty", "Unknown")
            difficulties.append(difficulty)
            if difficulty not in per_difficulty_count.keys():
                per_difficulty_count[difficulty] = 0

            rum = compute_rumination(item['Output'])
            item['Rumination'] = rum
            rum_list.append(rum)
            rum_per_difficulty[difficulty].append(rum)

            pred_code = extract_answer(item['Output'], mode='codegen')
            if pred_code != '':
                num_valid_answer += 1
                per_difficulty_count[difficulty] += 1
            public_test_cases = json.loads(item.get("public_test_cases", "{}"))

            inputs, outputs = [], []
            for case in public_test_cases:
                inputs.append(case["input"])
                outputs.append(case["output"])

            sample = {
                "input_output": json.dumps({
                    "inputs": inputs,
                    "outputs": outputs
                }),
            }

            samples_list.append(sample)
            generations_list.append([pred_code])
            item['Pred_Answer'] = pred_code
            item['Question'] = input_prompt


        metrics, results, final_metadata = codegen_metrics(
            samples_list,
            generations_list,
            k_list=[1],
            num_process_evaluate=2,
            timeout=10,
            debug=False,
        )

        pass_at_1 = metrics.get('pass@1', 0.0)
        detail_pass_at_1 = metrics['detail']['pass@1']

        for item, pass1, res, meta in zip(filtered_data, detail_pass_at_1.values(), results.values(), final_metadata):
            item['Metrics'] = {'pass@1': pass1}
            item['Results'] = res
            item['Final_metadata'] = meta

        difficulty_metrics = defaultdict(list)
        for idx, difficulty in enumerate(difficulties):
            pass1 = detail_pass_at_1[idx]
            difficulty_metrics[difficulty].append(pass1)

        overall_metrics = {
            'pass@1': pass_at_1,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
            'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
        }
        if avg_tokens is not None:
            overall_metrics['avg_output_tokens'] = f'{avg_tokens:.1f}'

        rum_overall = _aggregate_rumination(rum_list)
        if rum_overall is not None:
            overall_metrics['rumination'] = rum_overall

        per_difficulty_metrics = {}
        for difficulty, passes in difficulty_metrics.items():
            avg_pass = np.mean(passes) if len(passes) > 0 else 0.0
            num_valid_answer = per_difficulty_count[difficulty]
            per_difficulty_metrics[difficulty] = {
                'pass@1': avg_pass,
                'num_valid_answer': f'{num_valid_answer} of {len(passes)}'
            }
            rum_dom = _aggregate_rumination(rum_per_difficulty.get(difficulty, []))
            if rum_dom is not None:
                per_difficulty_metrics[difficulty]['rumination'] = rum_dom

        final_metrics = {
            'overall': overall_metrics,
            'per_domain': per_difficulty_metrics
        }

    else:
        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
        num_valid_answer = 0
        rum_list = []

        domain_metrics = {}
        _DOMAIN_FIELD = {'gpqa': 'High-level domain', 'hle': 'category', 'browsecomp': 'problem_topic'}
        domain_field = _DOMAIN_FIELD.get(dataset_name)

        item_modes: list = [None] * len(filtered_data)
        judge_queue = []
        for i, (item, input_prompt, result) in enumerate(zip(filtered_data, input_list, output_list)):
            if type(result) == str:
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text
            rum = compute_rumination(item['Output'])
            item['Rumination'] = rum
            rum_list.append(rum)

            if dataset_name in ['gpqa', 'medmcqa']:
                labeled_answer = item["Correct Choice"]
                item_modes[i] = ('choose', labeled_answer)
            elif dataset_name == 'hle' and item.get('answer_type') == 'multipleChoice':
                item_modes[i] = ('choose', item.get('answer', ''))
            elif dataset_name in ('hle', 'browsecomp'):
                golden = item.get("golden_answers", item.get("answer", ""))
                correct_answer = golden[0] if isinstance(golden, list) else golden
                judge_queue.append({
                    'idx': i,
                    'question': item.get('Question', item.get('question', input_prompt)),
                    'correct_answer': correct_answer,
                    'response': _extract_judge_response_text(item['Output']),
                })
            elif dataset_name in ['math500', 'aime', 'amc']:
                labeled_answer = item["answer"]
                item_modes[i] = ('gen', labeled_answer)
            elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'ambigqa', 'popqa']:
                labeled_answer = item.get("golden_answers", item["answer"])
                if isinstance(labeled_answer, str):
                    labeled_answer = [labeled_answer]
                item_modes[i] = ('qa', labeled_answer)
            elif dataset_name in ['pubhealth']:
                labeled_answer = item["answer"]
                item_modes[i] = ('choose', labeled_answer)
            else:
                raise ValueError(f"Unknown dataset_name: {dataset_name}")

        judge_results = {}
        if judge_queue:
            import os
            import sys
            sys.path.insert(0, os.environ.get(
                "PEARL_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from llm_judge_grade import grade_batch
            print(f"[judge] grading {len(judge_queue)} free-form {dataset_name} items via LLM judge...")
            graded = grade_batch([
                {'question': q['question'], 'correct_answer': q['correct_answer'], 'response': q['response']}
                for q in judge_queue
            ])
            for q, g in zip(judge_queue, graded):
                judge_results[q['idx']] = g

        for i, (item, input_prompt) in enumerate(zip(filtered_data, input_list)):
            rum = item['Rumination']
            if item_modes[i] is None:
                g = judge_results[i]
                is_correct = 1 if g['correct'] else 0
                metric = {
                    'is_valid_answer': g['extracted_final_answer'] not in ('', 'None'),
                    'acc': is_correct, 'em': is_correct, 'f1': float(is_correct), 'math_equal': 0,
                    'judge_confidence': g['confidence'], 'judge_reasoning': g['reasoning'],
                }
                pred_answer = g['extracted_final_answer']
                mode = 'judge'
            else:
                mode, labeled_answer = item_modes[i]
                metric, pred_answer = evaluate_predictions(output=item['Output'], labeled_answer=labeled_answer, mode=mode)

            item['Pred_Answer'] = pred_answer
            item['Metrics'] = metric
            item['Question'] = input_prompt

            if mode == 'judge':
                my_method_valid = metric['is_valid_answer']
            else:
                my_method_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])
            avg_math.append(metric['math_equal'])

            if my_method_valid:
                num_valid_answer += 1

            if domain_field:
                domain = item.get(domain_field, "Unknown")
                if domain not in domain_metrics:
                    domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0, 'rum': []}
                domain_metrics[domain]['total_num'] += 1
                domain_metrics[domain]['em'].append(metric['em'])
                domain_metrics[domain]['acc'].append(metric['acc'])
                domain_metrics[domain]['f1'].append(metric['f1'])
                domain_metrics[domain]['math_equal'].append(metric['math_equal'])
                domain_metrics[domain]['rum'].append(rum)
                if my_method_valid:
                    domain_metrics[domain]['num_valid_answer'] += 1

        t = time.localtime()
        result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
        metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

        overall_results = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
            'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0.0,
            'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
            'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
        }
        if avg_tokens is not None:
            overall_results['avg_output_tokens'] = f'{avg_tokens:.1f}'

        rum_overall = _aggregate_rumination(rum_list)
        if rum_overall is not None:
            overall_results['rumination'] = rum_overall

        domain_avg_metrics = {}
        if domain_field:
            for dm, m in domain_metrics.items():
                domain_avg_metrics[dm] = {
                    'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                    'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                    'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                    'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                    'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}'
                }
                rum_dom = _aggregate_rumination(m.get('rum', []))
                if rum_dom is not None:
                    domain_avg_metrics[dm]['rumination'] = rum_dom

        final_metrics = {'overall': overall_results}
        if domain_field:
            final_metrics['per_domain'] = domain_avg_metrics

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'
    if apply_backoff:
        result_json_name = output_dir
        metrics_json_name = output_dir.replace('.json', '.metrics.backoff.json')

    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model outputs with optional backoff.")
    parser.add_argument('--output_path', type=str, required=True, help='Path to the model output JSON file.')
    parser.add_argument('--output_metrics_path', type=str, help='Path to save the evaluation metrics.')
    parser.add_argument('--apply_backoff', action='store_true', help='Enable backoff to normal outputs if main output is invalid.')
    args = parser.parse_args()

    output_path = args.output_path
    if args.output_metrics_path:
        output_metrics_path = args.output_metrics_path
    else:
        output_metrics_path = output_path.replace('.json', '.metrics.json')

    if 'gpqa' in output_path:
        dataset_name = 'gpqa'
        normal_output_path = './outputs/gpqa.qwq.direct/diamond.12.13,18:23.json'
        if 'extended' in output_path:
            normal_output_path = './outputs/gpqa.qwq.direct/extended.12.28,15:44.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/gpqa.qwen2.5-32b-instruct.direct/diamond.12.14,20:34.json'
    elif 'math500' in output_path:
        dataset_name = 'math500'
        normal_output_path = './outputs/math500.qwq.direct/test.12.13,18:26.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/math500.qwen2.5-32b-instruct.direct/test.12.15,10:43.json'
    elif 'aime' in output_path:
        dataset_name = 'aime'
        normal_output_path = './outputs/aime.qwq.direct/2024.12.13,19:36.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/aime.qwen2.5-32b-instruct.direct/test.12.14,20:28.json'
    elif 'amc' in output_path:
        dataset_name = 'amc'
        normal_output_path = './outputs/amc.qwq.direct/test.12.14,14:31.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/amc.qwen2.5-32b-instruct.direct/test.12.14,20:26.json'
    elif 'livecode' in output_path:
        dataset_name = 'livecode'
        normal_output_path = './outputs/livecode.qwq.direct/test.12.13,21:24.json'
        if 'qwq' not in output_path:
            normal_output_path = './outputs/runs.baselines/livecode.qwen2.5-32b-instruct.direct/test.12.14,20:32.json'
    elif 'nq' in output_path:
        dataset_name = 'nq'
        normal_output_path = './outputs/runs.qa/nq.qwq.direct/test.12.15,14:50.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'triviaqa' in output_path:
        dataset_name = 'triviaqa'
        normal_output_path = './outputs/runs.qa/triviaqa.qwq.direct/test.12.15,15:35.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'hotpotqa' in output_path:
        dataset_name = 'hotpotqa'
        normal_output_path = './outputs/runs.qa/hotpotqa.qwq.direct/test.12.15,14:52.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'musique' in output_path:
        dataset_name = 'musique'
        normal_output_path = './outputs/runs.qa/musique.qwq.direct/test.12.27,16:44.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'bamboogle' in output_path:
        dataset_name = 'bamboogle'
        normal_output_path = './outputs/runs.qa/bamboogle.qwq.direct/test.12.28,9:51.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif '2wiki' in output_path:
        dataset_name = '2wiki'
        normal_output_path = './outputs/runs.qa/2wiki.qwq.direct/test.12.15,15:32.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'medmcqa' in output_path:
        dataset_name = 'medmcqa'
        normal_output_path = './outputs/runs.qa/medmcqa.qwq.direct/test.12.15,16:57.json'
        if 'qwq' not in output_path:
            normal_output_path = ''
    elif 'pubhealth' in output_path:
        dataset_name = 'pubhealth'
        normal_output_path = './outputs/runs.qa/pubhealth.qwq.direct/test.12.15,20:32.json'
        if 'qwq' not in output_path:
            normal_output_path = ''

    with open(output_path, mode='r', encoding='utf-8') as file:
        data = json.load(file)

    with open(output_metrics_path, mode='r', encoding='utf-8') as file:
        metrics = json.load(file)

    if 'overall' in metrics:
        query_latency = metrics['overall']['query_latency']
        original_num_valid_answer = metrics['overall']['num_valid_answer']
        avg_output_tokens = metrics['overall'].get('avg_output_tokens')
    else:
        query_latency = metrics.get('query_latency', 'N/A')
        original_num_valid_answer = metrics.get('num_valid_answer', 'N/A')
        avg_output_tokens = metrics.get('avg_output_tokens')

    normal_data = None
    if args.apply_backoff:
        if not os.path.exists(normal_output_path):
            raise FileNotFoundError(f"Normal output file not found at: {normal_output_path}")
        with open(normal_output_path, mode='r', encoding='utf-8') as file:
            normal_data = json.load(file)

    if dataset_name != 'livecode':
        avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
        num_valid_answer = 0
        rum_list = []

        domain_metrics = {}

        for i, item in enumerate(data):
            if dataset_name in ['gpqa', 'medmcqa']:
                labeled_answer = item["Correct Choice"]
                domain = item.get("High-level domain", "Unknown")
                mode = 'choose'
            elif dataset_name == 'math500':
                labeled_answer = item["answer"]
                domain = item.get("level", "Unknown")
                mode = 'gen'
            elif dataset_name in ['aime', 'amc']:
                labeled_answer = item["answer"]
                mode = 'gen'
                domain = 'Unknown'
            elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'ambigqa', 'popqa']:
                labeled_answer = item.get("golden_answers", item["answer"])
                if isinstance(labeled_answer, str):
                    labeled_answer = [labeled_answer]
                mode = 'qa'
                domain = 'Unknown'
            elif dataset_name in ['pubhealth']:
                labeled_answer = item["answer"]
                mode = 'choose'
                domain = 'Unknown'
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

            output = item['Output']

            metric, pred_answer = evaluate_predictions(
                output=output, 
                labeled_answer=labeled_answer,
                mode=mode,
            )

            my_method_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

            if args.apply_backoff and not my_method_valid and normal_data is not None:
                normal_item = normal_data[i]
                if dataset_name in ['gpqa', 'medmcqa']:
                    normal_labeled_answer = normal_item["Correct Choice"]
                    normal_mode = 'choose'
                elif dataset_name == 'math500':
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'gen'
                elif dataset_name in ['aime', 'amc']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'gen'
                elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'ambigqa', 'popqa']:
                    normal_labeled_answer = normal_item.get("golden_answers", normal_item["answer"])
                    if isinstance(normal_labeled_answer, str):
                        normal_labeled_answer = [normal_labeled_answer]
                    normal_mode = 'qa'
                elif dataset_name in ['pubhealth']:
                    normal_labeled_answer = normal_item["answer"]
                    normal_mode = 'choose'
                else:
                    raise ValueError(f"Unsupported dataset for backoff: {dataset_name}")

                normal_output = normal_item['Output']

                normal_metric, normal_pred_answer = evaluate_predictions(
                    output=normal_output, 
                    labeled_answer=normal_labeled_answer,
                    mode=normal_mode,
                )
                normal_valid = (normal_pred_answer != '' and not (normal_mode == 'choose' and dataset_name == 'gpqa' and len(normal_pred_answer) > 1))

                if normal_valid:
                    metric = normal_metric
                    pred_answer = normal_pred_answer
                    my_method_valid = True

            if domain not in domain_metrics:
                domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0, 'rum': []}
            domain_metrics[domain]['total_num'] += 1

            rum = compute_rumination(output)
            item['Rumination'] = rum
            rum_list.append(rum)
            domain_metrics[domain].setdefault('rum', []).append(rum)

            avg_em.append(metric['em'])
            avg_acc.append(metric['acc'])
            avg_f1.append(metric['f1'])
            avg_math.append(metric['math_equal'])
            domain_metrics[domain]['em'].append(metric['em'])
            domain_metrics[domain]['acc'].append(metric['acc'])
            domain_metrics[domain]['f1'].append(metric['f1'])
            domain_metrics[domain]['math_equal'].append(metric['math_equal'])

            if my_method_valid:
                num_valid_answer += 1
                domain_metrics[domain]['num_valid_answer'] += 1

        overall_metrics = {
            'em': np.mean(avg_em) if len(avg_em) > 0 else 0, 
            'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0, 
            'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0, 
            'math_equal': np.mean(avg_math) if len(avg_math) > 0 else 0, 
            'num_valid_answer': f'{num_valid_answer} of {len(data)}',
            'query_latency': query_latency,
        }
        if avg_output_tokens is not None:
            overall_metrics['avg_output_tokens'] = avg_output_tokens
        if args.apply_backoff:
            overall_metrics['original_num_valid_answer'] = original_num_valid_answer

        rum_overall = _aggregate_rumination(rum_list)
        if rum_overall is not None:
            overall_metrics['rumination'] = rum_overall

        domain_avg_metrics = {}
        for dm, m in domain_metrics.items():
            domain_avg_metrics[dm] = {
                'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}',
            }
            rum_dom = _aggregate_rumination(m.get('rum', []))
            if rum_dom is not None:
                domain_avg_metrics[dm]['rumination'] = rum_dom

        final_metrics = {'overall': overall_metrics}
        if dataset_name == 'gpqa':
            final_metrics['per_domain'] = domain_avg_metrics

    else:
        split = 'test'

        if args.apply_backoff and normal_data is not None:
            for i, item in enumerate(data):
                pred_answer = item['Pred_Answer']

                if pred_answer == '':
                    item['Output'] = normal_data[i]['Output']

        input_list = [item['Question'] for item in data]
        output_list = [item['Output'] for item in data]

        total_time = 0

        run_evaluation(
            filtered_data=data,
            input_list=input_list,
            output_list=output_list,
            dataset_name=dataset_name,
            output_dir=output_path,
            total_time=total_time,
            split=split,
            apply_backoff=True,
        )

    if dataset_name != 'livecode' or not args.apply_backoff:
        if args.apply_backoff:
            output_metrics_path = output_metrics_path.replace('.json', '.backoff.json')
        with open(output_metrics_path, mode='w', encoding='utf-8') as json_file:
            json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)

    print(f"Evaluation completed. Metrics saved to {output_metrics_path}")
