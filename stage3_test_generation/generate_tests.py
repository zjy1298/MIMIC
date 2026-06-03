#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test case generation script.
Generates test inputs for accepted_solutions in each enriched QA record,
for subsequent data synthesis. Structure based on instrument_code.py.
"""

import os
import time
import random
import json
import argparse

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import Dataset


def fix_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def load_prompt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


LANG_MAP = {
    'gnu c++': 'cpp', 'gnu c++11': 'cpp', 'gnu c++14': 'cpp',
    'gnu c++17': 'cpp', 'gnu c++20': 'cpp', 'ms c++': 'cpp',
    'c++': 'cpp', 'c': 'c',
    'python 2': 'python', 'python 3': 'python', 'python': 'python',
    'pypy 2': 'python', 'pypy 3': 'python', 'pypy': 'python',
    'pypy 3-64': 'python',
    'java': 'java', 'java 8': 'java', 'java 11': 'java',
    'kotlin': 'kotlin',
    'rust': 'rust', 'go': 'go',
    'javascript': 'javascript', 'node.js': 'javascript',
    'ruby': 'ruby', 'php': 'php',
}


def detect_language(prog_lang: str) -> str:
    key = prog_lang.strip().lower()
    if key in LANG_MAP:
        return LANG_MAP[key]
    for pattern, lang in LANG_MAP.items():
        if pattern in key:
            return lang
    return 'cpp'


# ──────────────────────────────────────────────────────────────
# Data loading and expansion
# ──────────────────────────────────────────────────────────────

def load_qa_data(input_path: str) -> list:
    data_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading enriched QA"):
            line = line.strip()
            if not line:
                continue
            data_list.append(json.loads(line))
    print(f"Loaded {len(data_list)} QA records")
    return data_list


def build_problem_text(qa: dict) -> str:
    problem_text = qa.get('refined_question', '') or qa.get('question', '')
    parts = [problem_text]
    if qa.get('input_format'):
        parts.append(f"\nInput format:\n{qa['input_format']}")
    if qa.get('output_format'):
        parts.append(f"\nOutput format:\n{qa['output_format']}")
    examples = qa.get('examples', [])
    if examples:
        parts.append("\nExamples:")
        for idx, ex in enumerate(examples):
            parts.append(
                f"  Example {idx+1}:\n"
                f"    Input:  {ex.get('input', '').strip()}\n"
                f"    Output: {ex.get('output', '').strip()}"
            )
    if qa.get('note'):
        parts.append(f"\nNote:\n{qa['note']}")
    return '\n'.join(parts)


def expand_solutions(
    qa_data: list,
    max_solutions_per_qa: int = None,
    max_code_len: int = None,
    allowed_languages: set = None,
) -> Dataset:
    records = []
    skip_lang = 0
    skip_len = 0
    skip_empty = 0

    for qa in tqdm(qa_data, desc="Expanding solutions"):
        solutions = qa.get('accepted_solutions', [])
        if not solutions:
            skip_empty += 1
            continue

        problem = build_problem_text(qa)

        valid_sols = []
        for sol in solutions:
            code = sol.get('source', '') or sol.get('og_source', '')
            if not code.strip():
                skip_empty += 1
                continue
            lang = detect_language(sol.get('programmingLanguage', ''))
            if allowed_languages and lang not in allowed_languages:
                skip_lang += 1
                continue
            if max_code_len and len(code) > max_code_len:
                skip_len += 1
                continue
            valid_sols.append((sol, code, lang))

        if not valid_sols:
            continue

        if max_solutions_per_qa and len(valid_sols) > max_solutions_per_qa:
            valid_sols = random.sample(valid_sols, max_solutions_per_qa)

        for sol_idx, (sol, code, lang) in enumerate(valid_sols):
            records.append({
                'id': qa.get('id', ''),
                'contest_id': str(qa.get('contest_id', '')),
                'index': qa.get('index', ''),
                'title': qa.get('title', ''),
                'type_level1': str(qa.get('type_level1', '')),
                'problem': problem,
                'code': code,
                'language': lang,
                'programmingLanguage': sol.get('programmingLanguage', ''),
                'submission_id': str(sol.get('submission_id', '')),
                'solution_idx': sol_idx,
            })

    print(f"Expanded to {len(records)} (qa, solution) pairs")
    print(f"  Skipped: empty={skip_empty}, lang_filter={skip_lang}, too_long={skip_len}")
    return Dataset.from_list(records)


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def build_chat_texts(
    examples: dict,
    prompt_template: str,
    system_prompt: str,
    tokenizer,
) -> list:
    chat_texts = []
    for problem, code, language in zip(
        examples['problem'],
        examples['code'],
        examples['language'],
    ):
        filled = (
            prompt_template
            .replace('{problem}', problem)
            .replace('{code}', code)
            .replace('{language}', language)
        )
        chat_texts.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": filled},
                ],
                tokenize=False,
                enable_thinking=False,
                add_generation_prompt=True,
            )
        )
    return chat_texts


def run_inference(args, tokenizer, llm, prompt_template, system_prompt,
                  sampling_params, shard_dataset) -> list:
    all_data = shard_dataset.to_dict()
    total = len(shard_dataset)
    results = []

    for start in tqdm(range(0, total, args.batch_size), desc="Inferencing"):
        end = min(start + args.batch_size, total)
        batch = {col: all_data[col][start:end] for col in all_data}
        chat_texts = build_chat_texts(
            batch, prompt_template, system_prompt, tokenizer,
        )
        outputs = llm.generate(chat_texts, sampling_params)
        results.extend(out.outputs[0].text.strip() for out in outputs)

    return results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args):
    fix_seed(args.seed)

    qa_data = load_qa_data(args.input_path)

    allowed_languages = None
    if args.allowed_languages:
        allowed_languages = set(args.allowed_languages)
        print(f"Filtering to languages: {allowed_languages}")

    dataset = expand_solutions(
        qa_data,
        max_solutions_per_qa=args.max_solutions_per_qa,
        max_code_len=args.max_code_len,
        allowed_languages=allowed_languages,
    )
    if len(dataset) == 0:
        print("Error: no solutions to process.")
        return

    prompt_template = load_prompt(args.prompt_path)
    system_prompt = (
        load_prompt(args.system_prompt_path) if args.system_prompt_path
        else "You are an algorithm test data construction expert. "
             "Your goal is to analyze code logic and generate test inputs "
             "that expose reasoning flaws, not computational limitations."
    )

    print(f"Loading tokenizer from '{args.model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side='left',
        use_fast=args.use_fast,
    )

    print(f"Loading model from '{args.model_name_or_path}'...")
    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        dtype='auto',
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )

    dataset = dataset.shard(args.shard[0], args.shard[1], contiguous=True)
    print(f"Using shard {args.shard[1]}/{args.shard[0]} ({len(dataset)} samples)")

    os.makedirs(args.output_path, exist_ok=True)
    total_samples = len(dataset)
    samples_per_shard = (total_samples + args.num_save_shards - 1) // args.num_save_shards

    start_time = time.time()

    for shard_idx in range(args.num_save_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx   = min((shard_idx + 1) * samples_per_shard, total_samples)
        if start_idx >= end_idx:
            continue

        print(f"\nProcessing save-shard {shard_idx}/{args.num_save_shards} "
              f"(samples {start_idx}-{end_idx})...")
        shard_ds = dataset.select(range(start_idx, end_idx))

        responses = run_inference(
            args, tokenizer, llm,
            prompt_template, system_prompt, sampling_params, shard_ds,
        )

        shard_output_path = os.path.join(args.output_path, f"shard_{shard_idx}")
        inferenced = shard_ds.add_column('generate_test_result', responses)
        print(f"Saving shard {shard_idx} to '{shard_output_path}'...")
        inferenced.save_to_disk(
            shard_output_path,
            num_proc=min(max(len(inferenced) - 1, 1), args.save_workers),
        )

    elapsed = time.time() - start_time
    print(f"\nDone. {total_samples} samples in {elapsed:.1f}s "
          f"({elapsed/max(total_samples,1):.3f}s/sample)")
    print(f"Output: {args.output_path}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate test inputs for AC code via LLM"
    )

    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True,
                        help="enriched QA jsonl (simplified_qa_enriched.jsonl)")
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True,
                        help="generate_test_en.txt")
    parser.add_argument('--system_prompt_path', type=str, default=None)

    parser.add_argument('--max_solutions_per_qa', type=int, default=None,
                        help="Max solutions per QA to generate tests for (None = all)")
    parser.add_argument('--max_code_len', type=int, default=8000,
                        help="Skip solutions longer than this (chars)")
    parser.add_argument('--allowed_languages', type=str, nargs='+', default=None,
                        help="Only process these languages, e.g. cpp python")

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_tokens', type=int, default=16384)
    parser.add_argument('--max_model_len', type=int, default=16384)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--shard', type=int, nargs=2, default=[1, 0],
                        help="[num_shards, shard_index]")
    parser.add_argument('--num_save_shards', type=int, default=1)
    parser.add_argument('--use_fast', action='store_true')
    parser.add_argument('--save_workers', type=int, default=4)

    args = parser.parse_args()
    print("args:", args)
    main(args)
