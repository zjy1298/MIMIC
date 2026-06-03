#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refine_synthesis.py
Refines synthesized problems to eliminate the "two-segment splicing" issue.
Reads HF Dataset from synthesized_typed_new and uses a refine prompt to have the model rewrite problems.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python refine_synthesis.py \
      --model_name_or_path ./models/model_checkpoint \
      --input_base ./data/input.jsonl \
      --output_path ./data/input.jsonl \
      --prompt_path ./prompts/refine_synthesis.txt \
      --shard 40 0 \
      --batch_size 200
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


def load_synthesized_data(input_base: str) -> Dataset:
    shard_dirs = sorted(glob.glob(os.path.join(input_base, "shard_*")))
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* dirs found in {input_base}")

    datasets = []
    for shard_dir in tqdm(shard_dirs, desc="Loading shards"):
        inner = os.path.join(shard_dir, "shard_0")
        path = inner if os.path.isdir(inner) else shard_dir
        try:
            ds = load_from_disk(path)
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: skip {shard_dir}: {e}")

    combined = concatenate_datasets(datasets)
    print(f"Loaded {len(combined)} records from {len(datasets)} shards")
    return combined


def extract_draft_and_input(synthesis_result: str):
    try:
        obj = json.loads(synthesis_result)
        return obj.get('problem', ''), obj.get('input', '')
    except (json.JSONDecodeError, TypeError):
        return synthesis_result, ''


def build_chat_texts(examples: dict, prompt_template: str, system_prompt: str,
                     tokenizer) -> list:
    chat_texts = []
    for synthesis_result, corpus_text in zip(
        examples['synthesis_result'],
        examples['corpus_text'],
    ):
        draft_problem, draft_input = extract_draft_and_input(synthesis_result)

        filled = (
            prompt_template
            .replace('{draft_problem}', draft_problem)
            .replace('{input}', draft_input)
            .replace('{corpus}', corpus_text)
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
            batch, prompt_template, system_prompt, tokenizer
        )
        outputs = llm.generate(chat_texts, sampling_params)
        results.extend(out.outputs[0].text.strip() for out in outputs)

    return results


def main(args):
    fix_seed(args.seed)

    dataset = load_synthesized_data(args.input_base)
    if len(dataset) == 0:
        print("Error: no data to refine.")
        return

    prompt_template = load_prompt(args.prompt_path)
    system_prompt = "You are an expert exam question editor who specializes in making questions read naturally and coherently."

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
        inferenced = shard_ds.add_column('refined_result', responses)
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
        description="Refine synthesized problems to improve narrative coherence"
    )

    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--input_base', type=str, required=True,
                        help="synthesized_typed_new directory (containing shard_* subdirectories)")
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True,
                        help="Path to refine_synthesis.txt")

    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--max_tokens', type=int, default=2048)
    parser.add_argument('--max_model_len', type=int, default=16384)
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
