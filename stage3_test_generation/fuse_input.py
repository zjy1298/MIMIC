#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuse_input.py
Fuses official/generated test case inputs into the story narratives of synthesized problems.
Selects one synthesized story template per problem, and for each test case, calls the LLM
to integrate the new input into the narrative.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python fuse_input.py \
      --model_name_or_path ./models/model_checkpoint \
      --synthesized_dir ./data/input.jsonl \
      --eval_data ./data/input.jsonl \
      --output_path ./data/input.jsonl \
      --prompt_path ./prompts/fuse_input.txt \
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
from datasets import Dataset, load_from_disk


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


def load_story_templates(synthesized_dir: str) -> dict:
    """Load all story templates per problem ID from synthesized data.
    Returns {pid: [(problem_text, original_input), ...]}."""
    templates = {}
    shard_dirs = sorted(glob.glob(os.path.join(synthesized_dir, 'shard_*')))

    for sd in shard_dirs:
        inner = os.path.join(sd, 'shard_0')
        path = inner if os.path.isdir(inner) else sd
        try:
            ds = load_from_disk(path)
        except:
            continue

        for r in ds:
            pid = r['id']
            for field in ['refined_result', 'synthesis_result']:
                try:
                    obj = json.loads(r[field])
                    p = obj.get('problem', '')
                    i = obj.get('input', '')
                    if p and i is not None:
                        if pid not in templates:
                            templates[pid] = []
                        templates[pid].append((p, i))
                        break
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

    total_templates = sum(len(v) for v in templates.values())
    print(f"Loaded {total_templates} story templates for {len(templates)} problems "
          f"(avg {total_templates/max(len(templates),1):.1f} per problem)")
    return templates


def load_eval_cases(eval_data_path: str) -> list:
    """Load eval cases (official + generated test cases)."""
    records = []
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading eval cases"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(d)
            except:
                continue
    print(f"Loaded {len(records)} eval cases")
    return records


def build_fuse_dataset(eval_cases: list, templates: dict) -> Dataset:
    """Match eval cases with story templates, rotating through available templates."""
    rows = []
    skipped = 0
    pid_counters = {}
    for case in eval_cases:
        pid = case.get('problem_id', '')
        if pid not in templates or not templates[pid]:
            skipped += 1
            continue

        tpl_list = templates[pid]
        idx = pid_counters.get(pid, 0)
        story_problem, story_input = tpl_list[idx % len(tpl_list)]
        pid_counters[pid] = idx + 1

        rows.append({
            'eval_id': case.get('eval_id', ''),
            'problem_id': pid,
            'title': case.get('title', ''),
            'source': case.get('source', ''),
            'type_id': case.get('type_id', 0),
            'rating': case.get('rating', 0),
            'story_problem': story_problem,
            'original_input': story_input,
            'new_input': case.get('input', ''),
            'expected_output': case.get('expected_output', ''),
            'template_idx': idx % len(tpl_list),
        })

    print(f"Built {len(rows)} fuse pairs (skipped {skipped} without template)")
    return Dataset.from_list(rows)


def build_chat_texts(examples: dict, prompt_template: str, system_prompt: str,
                     tokenizer) -> list:
    chat_texts = []
    for story, orig_inp, new_inp in zip(
        examples['story_problem'],
        examples['original_input'],
        examples['new_input'],
    ):
        filled = (
            prompt_template
            .replace('{problem}', story)
            .replace('{original_input}', orig_inp)
            .replace('{new_input}', new_inp)
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

    templates = load_story_templates(args.synthesized_dir)
    eval_cases = load_eval_cases(args.eval_data)
    dataset = build_fuse_dataset(eval_cases, templates)

    if len(dataset) == 0:
        print("Error: no fuse pairs to process.")
        return

    prompt_template = load_prompt(args.prompt_path)
    system_prompt = "You are an expert exam question editor who seamlessly integrates new data into existing narratives."

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
        inferenced = shard_ds.add_column('fused_result', responses)
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
        description="Fuse test case inputs into story-based exam questions"
    )

    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--synthesized_dir', type=str, required=True,
                        help="synthesized_refined_v2 directory")
    parser.add_argument('--eval_data', type=str, required=True,
                        help="eval_data_v2.jsonl (official + generated test cases)")
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)

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
