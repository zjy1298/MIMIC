#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthesize_cot.py
Synthesizes Chain of Thought using an LLM: given a story-based problem statement +
correct answer + TRACE execution log, generates a natural language reasoning chain
described in the story context.

Data sources:
  - fused_eval: story-based problem statement + expected_output
  - trace_outputs: TRACE intermediate computation steps

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python synthesize_cot.py \
      --model_name_or_path ./models/model_checkpoint \
      --fused_eval_dir ./data/input.jsonl \
      --trace_dir ./data/input.jsonl \
      --output_path ./data/input.jsonl \
      --prompt_path ./prompts/synthesize_cot.txt \
      --shard 80 0 \
      --batch_size 100
"""

import os
import time
import random
import json
import argparse
import glob

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import Dataset, load_from_disk, concatenate_datasets


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


def load_traces(trace_dir: str, instrumented_dir: str) -> dict:
    """Load raw execution outputs indexed by (problem_id, source, input_hash).
    Returns {key: raw_stdout}.
    Also loads instrumented code per problem_id: {pid: instrumented_code}."""
    trace_map = {}
    files = sorted(glob.glob(os.path.join(trace_dir, 'traces_job*_s*.jsonl')))
    if not files:
        files = sorted(glob.glob(os.path.join(trace_dir, 'traces_job*.jsonl')))

    for fpath in tqdm(files, desc="Loading traces"):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                except:
                    continue
                raw = d.get('raw_stdout', '')
                if not raw.strip():
                    continue
                pid = d.get('problem_id', '')
                source = d.get('source', '')
                inp_hash = hash(d.get('input', '').replace('\r', '').strip())
                key = (pid, source, inp_hash)
                if key not in trace_map:
                    trace_map[key] = raw

    print(f"Loaded raw outputs for {len(trace_map)} unique (pid, source, input) combos")

    # Load instrumented code per problem
    inst_codes = {}
    shard_dirs = sorted(glob.glob(os.path.join(instrumented_dir, 'shard_*')))
    for sd in shard_dirs:
        inner = os.path.join(sd, 'shard_0')
        path = inner if os.path.isdir(inner) else sd
        try:
            ds = load_from_disk(path)
        except:
            continue
        for r in ds:
            pid = r['id']
            if pid in inst_codes:
                continue
            try:
                result = json.loads(r['instrument_result'])
                code = result.get('instrumented_code', '')
                if code:
                    inst_codes[pid] = code
            except:
                continue

    print(f"Loaded instrumented code for {len(inst_codes)} problems")
    return trace_map, inst_codes


def load_fused_eval(fused_eval_dir: str) -> Dataset:
    """Load fused eval data from shard directories."""
    shard_dirs = sorted(glob.glob(os.path.join(fused_eval_dir, 'shard_*')))
    datasets = []
    for sd in tqdm(shard_dirs, desc="Loading fused eval"):
        inner = os.path.join(sd, 'shard_0')
        path = inner if os.path.isdir(inner) else sd
        try:
            ds = load_from_disk(path)
            datasets.append(ds)
        except:
            continue
    combined = concatenate_datasets(datasets)
    print(f"Loaded {len(combined)} fused eval records")
    return combined


def build_cot_dataset(fused_ds: Dataset, trace_map: dict, inst_codes: dict) -> Dataset:
    """Match fused eval records with raw execution outputs and instrumented code."""
    rows = []
    matched = 0
    no_trace = 0

    for r in tqdm(fused_ds, desc="Matching traces"):
        try:
            fused = json.loads(r['fused_result'])
            problem = fused.get('problem', '')
        except:
            continue

        if not problem:
            continue

        pid = r['problem_id']
        source = r['source']
        inp = r['new_input'].replace('\r', '').strip()
        inp_hash = hash(inp)
        key = (pid, source, inp_hash)

        raw_stdout = trace_map.get(key, '')
        if not raw_stdout.strip():
            no_trace += 1
            continue

        instrumented_code = inst_codes.get(pid, '')
        if not instrumented_code:
            no_trace += 1
            continue

        matched += 1
        rows.append({
            'eval_id': r['eval_id'],
            'problem_id': pid,
            'title': r['title'],
            'source': source,
            'type_id': r.get('type_id', 0),
            'rating': r.get('rating', 0),
            'problem': problem,
            'answer': r['expected_output'],
            'raw_stdout': raw_stdout,
            'instrumented_code': instrumented_code,
        })

    print(f"Built {len(rows)} CoT records (matched: {matched}, skipped: {no_trace})")
    return Dataset.from_list(rows)


def build_chat_texts(examples: dict, prompt_template: str, system_prompt: str,
                     tokenizer) -> list:
    chat_texts = []
    for problem, answer, instrumented_code, raw_stdout in zip(
        examples['problem'],
        examples['answer'],
        examples['instrumented_code'],
        examples['raw_stdout'],
    ):
        filled = (
            prompt_template
            .replace('{problem}', problem)
            .replace('{answer}', answer)
            .replace('{instrumented_code}', instrumented_code)
            .replace('{raw_stdout}', raw_stdout)
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
    max_input_len = args.max_model_len - args.max_tokens
    skipped = 0

    for start in tqdm(range(0, total, args.batch_size), desc="Inferencing"):
        end = min(start + args.batch_size, total)
        batch = {col: all_data[col][start:end] for col in all_data}
        chat_texts = build_chat_texts(
            batch, prompt_template, system_prompt, tokenizer
        )

        # Filter out prompts that exceed max input length
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(chat_texts):
            token_len = len(tokenizer.encode(text, add_special_tokens=False))
            if token_len <= max_input_len:
                valid_indices.append(i)
                valid_texts.append(text)
            else:
                skipped += 1

        # Run inference on valid prompts
        batch_results = [''] * len(chat_texts)
        if valid_texts:
            outputs = llm.generate(valid_texts, sampling_params)
            for idx, out in zip(valid_indices, outputs):
                batch_results[idx] = out.outputs[0].text.strip()

        results.extend(batch_results)

    if skipped > 0:
        print(f"Skipped {skipped} prompts exceeding max input length ({max_input_len} tokens)")
    return results


def main(args):
    fix_seed(args.seed)

    trace_map, inst_codes = load_traces(args.trace_dir, args.instrumented_dir)
    fused_ds = load_fused_eval(args.fused_eval_dir)
    dataset = build_cot_dataset(fused_ds, trace_map, inst_codes)

    if len(dataset) == 0:
        print("Error: no CoT records to process.")
        return

    prompt_template = load_prompt(args.prompt_path)
    system_prompt = "You are an expert reasoning assistant who explains solutions using natural, story-appropriate language."

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
        end_idx = min((shard_idx + 1) * samples_per_shard, total_samples)
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
        inferenced = shard_ds.add_column('cot_result', responses)
        print(f"Saving shard {shard_idx} to '{shard_output_path}'...")
        inferenced.save_to_disk(
            shard_output_path,
            num_proc=min(max(len(inferenced) - 1, 1), args.save_workers),
        )

    elapsed = time.time() - start_time
    print(f"\nDone. {total_samples} samples in {elapsed:.1f}s "
          f"({elapsed/max(total_samples,1):.3f}s/sample)")
    print(f"Output: {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Synthesize Chain of Thought from fused problems + traces"
    )

    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--fused_eval_dir', type=str, required=True)
    parser.add_argument('--trace_dir', type=str, required=True)
    parser.add_argument('--instrumented_dir', type=str, required=True,
                        help="instrumented_code_new directory")
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--max_model_len', type=int, default=32768)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--shard', type=int, nargs=2, default=[1, 0],
                        help="[num_shards, shard_index]")
    parser.add_argument('--num_save_shards', type=int, default=1)
    parser.add_argument('--use_fast', action='store_true')
    parser.add_argument('--save_workers', type=int, default=4)

    args = parser.parse_args()
    print("args:", args)
    main(args)
