#!/usr/bin/env python3
"""
Calculate the accuracy of the `results.jsonl` file for each primary subject
and the overall accuracy.

Two types of accuracy are computed:
1. overall: based on the `correct` field in the results file
2. result: by comparing the `result`/`verdict` field in the results file
   with the `result` field in the benchmark file

Special file handling:
- For xverify, compass, tencent, and general verifiers, there is no `result`
  field, only a `verdict` field
- The `verdict` field needs to be compared with the `result` in the benchmark
  file

Data matching:
- Each data entry is uniquely identified by the `idx` field
- If the results file does not contain a line from the benchmark file, it is
  counted as incorrect
"""

import json
import os
import glob
from collections import defaultdict

def load_jsonl_file(file_path):
    """Load a jsonl file into a dict keyed by `idx`."""
    data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    idx = item.get('idx')
                    if idx:
                        data[idx] = item
                except json.JSONDecodeError:
                    continue
    return data


def calculate_scores(benchmark_data, result_data, result_field, specialized_verifier=False):
    """
    Calculate accuracy and confusion matrix statistics.

    Args:
        benchmark_data: benchmark data (dict, key=idx)
        result_data: result data (dict, key=idx)
        result_field: field name used for comparison ('result' or 'verdict')
        specialized_verifier: whether this is a specialized verifier (e.g. xverify, compassverifier)

    Returns:
        dict: per-subject accuracy and confusion matrix statistics
    """
    subject_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})

    for idx, benchmark_item in benchmark_data.items():
        subject = benchmark_item.get('first_level_subject', 'unknown')
        subject_stats[subject]['total'] += 1

        if idx not in result_data:
            continue

        result_item = result_data[idx]
        # Specialized verifiers only have `verdict` field
        if specialized_verifier:
            verdict = result_item.get('verdict')
            if verdict is None:
                verdict = 0
            expected = benchmark_item.get(result_field)
            # `verdict` may be bool or int, need to normalize
            if isinstance(verdict, bool):
                verdict = 1 if verdict else 0
            if verdict == 1 and expected == 1:
                subject_stats[subject]['TP'] += 1
                subject_stats[subject]['correct'] += 1
            elif verdict == 1 and expected == 0:
                subject_stats[subject]['FP'] += 1
            elif verdict == 0 and expected == 1:
                subject_stats[subject]['FN'] += 1
            elif verdict == 0 and expected == 0:
                subject_stats[subject]['TN'] += 1
                subject_stats[subject]['correct'] += 1
            else:
                raise ValueError(f"Invalid verdict or expected value: verdict={verdict}, expected={expected}")
        else:
            # General LLM files have both `result` and `verdict` fields
            if result_field == 'result':
                verdict = result_item.get('result')
            elif result_field == 'verdict':
                verdict = result_item['process'] * result_item['result']
            else:
                raise ValueError(f"Invalid result field: {result_field}")
            expected = benchmark_item.get(result_field)
            if verdict == 1 and expected == 1:
                subject_stats[subject]['TP'] += 1
                subject_stats[subject]['correct'] += 1
            elif verdict == 1 and expected == 0:
                subject_stats[subject]['FP'] += 1
            elif verdict == 0 and expected == 1:
                subject_stats[subject]['FN'] += 1
            elif verdict == 0 and expected == 0:
                subject_stats[subject]['TN'] += 1
                subject_stats[subject]['correct'] += 1
            else:
                print(result_item)
                raise ValueError(f"Invalid verdict or expected value: verdict={verdict}, expected={expected}")
        # If the results file does not contain this idx, it is counted as incorrect

    return subject_stats


def print_accuracy_report(stats, title):
    """Print accuracy and overall metrics report."""
    print(f"{title}\n")

    total_correct = 0
    total_count = 0
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0

    for subject in ['math', 'physics', 'chemistry', 'biology']:
        data = stats[subject]
        count = data['total']
        correct = data['correct']
        accuracy = correct / count if count > 0 else 0
        print(f"{subject}: {correct}/{count} ({accuracy:.4f})")

        total_correct += correct
        total_count += count
        total_TP += data['TP']
        total_FP += data['FP']
        total_FN += data['FN']
        total_TN += data['TN']
        
    print()
    valid_count = total_TP + total_FP + total_FN + total_TN
    overall_accuracy = total_correct / total_count if total_count > 0 else 0
    overall_f1_score = 2 * total_TP / (2 * total_TP + total_FP + total_FN) if total_TP > 0 else 0
    print(f"Overall Accuracy: {total_correct}/{total_count} ({overall_accuracy:.4f})")
    print(f"Overall F1 Score: {overall_f1_score:.4f}")
    print(f"Overall TP: {total_TP / valid_count:.4f}, FP: {total_FP / valid_count:.4f}, FN: {total_FN / valid_count:.4f}, TN: {total_TN / valid_count:.4f}")
    # print(f"Overall TP: {total_TP}, FP: {total_FP}, FN: {total_FN}, TN: {total_TN}")

def calculate_token_number(result_data):
    """Calculate the average token number from result data."""
    token_num = []
    for item in result_data.values():
        if 'token_num' not in item:
            return 0
        token_num.append(item['token_num'])
    return sum(token_num) / len(token_num)

def main(
    benchmark_file,
    result_file,
    result_field,
    specialized_verifier=False,
):
    # Load benchmark data
    benchmark_data = load_jsonl_file(benchmark_file)
    print(f"Loaded {len(benchmark_data)} benchmark entries")

    # Load result data
    result_data = load_jsonl_file(result_file)
    print(f"Loaded {len(result_data)} result entries.")

    result_stats = calculate_scores(benchmark_data, result_data, result_field, specialized_verifier)
    print_accuracy_report(result_stats, f"{os.path.basename(result_file)}-{result_field}")

    token_num = calculate_token_number(result_data)
    print(f"Token number: {token_num}")
    print("-" * 50)


if __name__ == '__main__':
    # Benchmark file path
    benchmark_file = './data/PRIME.jsonl'
    # Result file path
    result_file = './results/EVALUATED_MODEL_results.jsonl'

    main(
        benchmark_file,
        result_file,
        result_field='verdict',  # 'verdict' for overall accuracy, 'result' for outcome accuracy
        specialized_verifier=False,  # True for specialized verifier, False for general LLM
    )
