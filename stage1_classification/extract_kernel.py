#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cognitive task classification annotation for Codeforces/LeetCode problems using vLLM.
Based on the classification framework in classify.txt, categorizes problems into 5 major classes and 15 subclasses.
"""

import os
import time
import argparse
import functools
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset, concatenate_datasets
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import random
import json
from datasets import Dataset

def fix_seed(seed=42):
    """Ensure reproducibility of results."""
    assert isinstance(seed, int)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def load_prompt(prompt_path):
    """Load a prompt file."""
    with open(prompt_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_jsonl_dataset(file_paths):
    """Load JSONL files and convert to a Dataset."""
    data_list = []
    for file_path in file_paths:
        print(f"Loading {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Reading {os.path.basename(file_path)}"):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                data_list.append(data)

    return Dataset.from_list(data_list)


def inference_worker(args, tokenizer, llm, prompt, system_prompt, sampling_params, examples):
    """Batch inference worker (called directly in main process, bypasses map pickle)."""
    chat_texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt.replace('{question}', question)}
            ],
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True
        )
        for question in examples['question']
    ]

    outputs = llm.generate(chat_texts, sampling_params)
    examples[args.output_field] = [output.outputs[0].text.strip() for output in outputs]
    return examples


def run_inference_on_dataset(args, tokenizer, llm, prompt_template, system_prompt, sampling_params, shard_dataset):
    """
    Run batch inference directly in the main process, completely bypassing
    the pickle serialization issue with datasets.map.
    """
    all_data = shard_dataset.to_dict()
    total = len(shard_dataset)
    batch_size = args.batch_size
    result_responses = []

    for start in tqdm(range(0, total, batch_size), desc="Inferencing"):
        end = min(start + batch_size, total)
        batch_examples = {col: all_data[col][start:end] for col in all_data}

        batch_result = inference_worker(
            args, tokenizer, llm,
            prompt_template, system_prompt, sampling_params,
            batch_examples
        )
        result_responses.extend(batch_result[args.output_field])

    inferenced_shard = shard_dataset.add_column(args.output_field, result_responses)
    return inferenced_shard


def main(args):
    fix_seed(args.seed)

    print(f"Loading '{args.input_path}'...")

    if args.jsonl:
        dataset = load_jsonl_dataset(args.input_path)
    elif args.json:
        data_array = [
            load_dataset("json", data_files=path, split="train")
            for path in tqdm(args.input_path, desc="Loading JSON files")
        ]
        dataset = concatenate_datasets(data_array)
    else:
        dataset = concatenate_datasets([
            load_from_disk(path) for path in tqdm(args.input_path, desc="Loading datasets")
        ])

    print(f"Loaded dataset with {len(dataset)} samples")

    prompt_template = load_prompt(args.prompt_path)
    system_prompt = load_prompt(args.system_prompt_path) if args.system_prompt_path else "You are a cognitive task analysis expert."

    print(f"Loading tokenizer '{args.model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        use_fast=args.use_fast
    )

    if args.gpu_rank:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, args.gpu_rank))

    print(f"Loading model '{args.model_name_or_path}'...")
    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        dtype='auto',
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        max_model_len=args.max_model_len,
    )
    print(f"Loaded model")

    if 'Qwen' in args.model_name_or_path or 'qwen' in args.model_name_or_path.lower():  # model-specific chat template
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
            seed=args.seed,
            max_tokens=args.max_tokens
        )
    else:
        sampling_params = SamplingParams(
            temperature=args.temperature,
            seed=args.seed,
            max_tokens=args.max_tokens
        )

    start_time = time.time()

    if not args.debug_prompt:
        dataset = dataset.shard(args.shard[0], args.shard[1], contiguous=True)
        print(f"Using shard {args.shard[1]}/{args.shard[0]} ({len(dataset)} samples)")
    else:
        num_segments = len(dataset) // 500 + 1
        sampled_indices = []
        for i in range(num_segments):
            start_idx = i * (len(dataset) // num_segments)
            end_idx = (i + 1) * (len(dataset) // num_segments)
            segment_indices = list(range(start_idx, min(end_idx, len(dataset))))
            sampled_indices.extend(random.sample(segment_indices, min(500 // num_segments, len(segment_indices))))
        dataset = dataset.select(sampled_indices)
        print(f"Debug mode: using {len(dataset)} samples")

    os.makedirs(args.output_path, exist_ok=True)

    total_samples = len(dataset)
    samples_per_shard = (total_samples + args.num_save_shards - 1) // args.num_save_shards

    print(f"Total samples: {total_samples}, Samples per shard: {samples_per_shard}, Number of shards: {args.num_save_shards}")

    for shard_idx in range(args.num_save_shards):
        shard_output_path = os.path.join(args.output_path, f"shard_{shard_idx}")

        start_idx = shard_idx * samples_per_shard
        end_idx = min((shard_idx + 1) * samples_per_shard, total_samples)

        if start_idx >= end_idx:
            print(f"Shard {shard_idx} is empty, skipping.")
            continue

        print(f"\nProcessing shard {shard_idx}/{args.num_save_shards} (samples {start_idx} to {end_idx})...")
        shard_dataset = dataset.select(range(start_idx, end_idx))

        # Direct batch inference, bypasses datasets.map
        print(f"Running inference on shard {shard_idx}...")
        inferenced_shard = run_inference_on_dataset(
            args, tokenizer, llm,
            prompt_template, system_prompt, sampling_params,
            shard_dataset
        )

        print(f"Saving shard {shard_idx} to '{shard_output_path}'...")
        inferenced_shard.save_to_disk(
            shard_output_path,
            num_proc=min(len(inferenced_shard) - 1, args.save_workers)
        )
        print(f"Shard {shard_idx} saved to '{shard_output_path}'")

    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Finished all inference in {elapsed_time:.2f}s")
    print(f"All shards saved to '{args.output_path}'")
    print(f"Average time per sample: {elapsed_time / max(total_samples, 1):.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-based cognitive task classification for Codeforces/LeetCode problems")

    # Model
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Model and tokenizer path')

    # Data
    parser.add_argument("--input_path", type=str, nargs="+", required=True, help="Input data path(s)")
    parser.add_argument('--output_path', type=str, required=True, help='Output data path')
    parser.add_argument("--json", action="store_true", help="Input is JSON dataset")
    parser.add_argument("--jsonl", action="store_true", help="Input is JSONL dataset")

    # Prompt
    parser.add_argument('--prompt_path', type=str, required=True, help='Prompt template path')
    parser.add_argument('--system_prompt_path', type=str, default=None, help='System prompt path')
    parser.add_argument("--input_field", type=str, default="question", help="Input field name")
    parser.add_argument("--output_field", type=str, default="llm_response", help="Output field name")

    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--max_tokens', type=int, default=1024, help='Maximum number of output tokens')
    parser.add_argument('--max_model_len', type=int, default=131072, help='Maximum model length')
    parser.add_argument('--temperature', type=float, default=0.0, help='Generation temperature')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Sharding parameters
    parser.add_argument("--shard", type=int, nargs=2, default=[1, 0], help="Dataset shard [num_shards, shard_index]")
    parser.add_argument("--num_save_shards", type=int, default=1, help="Number of shards to save output")
    parser.add_argument("--gpu_rank", type=int, nargs="+", default=None, help="CUDA_VISIBLE_DEVICES")

    # Misc
    parser.add_argument("--use_fast", action="store_true", help="Use fast tokenizer")
    parser.add_argument("--save_workers", type=int, default=4, help='Number of huggingface save processes')
    parser.add_argument("--debug_prompt", action="store_true", help="Debug mode: use only 500 samples")

    args = parser.parse_args()

    print("="*50)
    print("MIMIC - Cognitive Task Classification")
    print("="*50)
    print("args:", args)
    print("="*50)

    main(args)
