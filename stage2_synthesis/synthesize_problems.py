#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Type-aware contextual problem synthesis script.
Type-aware problem synthesis with additions:
  - Buckets classified corpus by suitable_types
  - Matches QA data to same-type corpus by type_level1
  - Uses synthesis_typed.txt, injecting type-specific fusion strategies
"""

import os
import time
import random
import json
import argparse
from collections import defaultdict

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import Dataset


# ──────────────────────────────────────────────────────────────
# Fusion strategy description for each type (injected into prompt as {type_instruction})
# ──────────────────────────────────────────────────────────────

TYPE_INSTRUCTIONS = {
    1: """\
**Type 1 – Visual/Spatial Reasoning**

The corpus describes spatial relationships, directions, positions, or a map-like layout.

Fusion strategy:
- Use the corpus location/entity names (buildings, rooms, regions, cells) as the objects in the spatial problem.
- Re-express the grid, map, or coordinate structure using the corpus's own terminology. For example, if the corpus mentions "north entrance", "east wing", "corridor 3B", use those as grid cell labels.
- The exam question should read like a floor-plan analysis or navigation task in the context described by the corpus.
- The `input` field must encode the spatial structure in the exact format expected by the original code (e.g., grid rows, coordinate pairs).""",

    2: """\
**Type 2 – Character-Level Manipulation**

The corpus passage itself is the raw material to be processed. The exam question asks the test-taker to perform character-level operations directly on a fragment of the corpus text.

Fusion strategy:
- Select a representative excerpt (1–4 sentences) from the corpus as the string to be operated on.
- Frame the question as a text analysis task in the domain of the corpus (e.g., "In the following excerpt from a maritime policy report, count how many times the letter 'a' appears in the word 'navigation'").
- The `input` field must contain the actual text excerpt (or derived string) followed by any query parameters, formatted exactly as the original code expects.
- Do NOT paraphrase or summarize the excerpt — use verbatim text so the answer is deterministic.""",

    3: """\
**Type 3 – Precise Mathematical Computation**

The corpus contains concrete numeric values (measurements, statistics, prices, counts, dates, percentages). These numbers become the parameters of the calculation problem.

Fusion strategy:
- Extract at least 2 specific numeric values from the corpus and use them as the problem's given quantities.
- Frame the calculation within the corpus's domain (e.g., "According to the report, trade volume is $5 trillion and the sea area is 3.5 million km²…").
- The narrative context (who, where, why) must come from the corpus — the numbers must also come from the corpus, not be invented.
- The `input` field must list the extracted numeric values in the exact format expected by the original code.""",

    4: """\
**Type 4 – Data Processing**

The corpus contains a structured list, table, or enumeration of entities with multiple attributes. This structured data becomes the dataset for the problem.

Fusion strategy:
- Identify the structured entries in the corpus (e.g., a list of regulations, a table of measurements, a ranked list of countries).
- Frame the problem as a data analysis task: filtering, sorting, grouping, or aggregating those entries.
- Present the data naturally in the exam question (as a table or bulleted list) within the corpus's narrative context.
- The `input` field must encode the structured data in the exact format expected by the original code (e.g., number of entries followed by attribute rows).""",

    5: """\
**Type 5 – State/Rule Simulation**

The corpus describes a step-by-step process, iterative procedure, lifecycle, or rule-based system. These rules become the simulation to execute.

Fusion strategy:
- Map the corpus's described process onto the algorithmic state transitions. Use the corpus's own terminology for states, transitions, and stopping conditions.
- Frame the problem as "given this process and an initial situation, what is the outcome after N steps / when does the system reach state X?"
- The initial state and any parameters should be derived from numeric or categorical values in the corpus where possible.
- The `input` field must encode the initial state and parameters in the exact format expected by the original code.""",
}

TYPE_INSTRUCTIONS_ZH = {
    1: """\
**Type 1 – Visual/Spatial Reasoning**

The corpus describes spatial relationships, directions, positions, or a map-like layout.

Fusion strategy:
- Use the corpus location/entity names (buildings, rooms, regions, cells) as the objects in the spatial problem.
- Re-express the grid, map, or coordinate structure using the corpus's own terminology. For example, if the corpus mentions "north entrance", "east wing", "corridor 3B", use those as grid cell labels.
- The exam question should read like a floor-plan analysis or navigation task in the context described by the corpus.
- The `input` field must encode the spatial structure in the exact format expected by the original code (e.g., grid rows, coordinate pairs).""",

    2: """\
**Type 2 – Character-Level Manipulation**

The corpus passage itself is the raw material to be processed. The exam question asks the test-taker to perform character-level operations directly on a fragment of the corpus text.

Fusion strategy:
- Select a representative excerpt (1-4 sentences) from the corpus as the string to be operated on.
- Frame the question as a text analysis task in the domain of the corpus (e.g., "In the following excerpt from a maritime policy report, count how many times the letter 'a' appears in the word 'navigation'").
- The `input` field must contain the actual text excerpt (or derived string) followed by any query parameters, formatted exactly as the original code expects.
- Do NOT paraphrase or summarize the excerpt -- use verbatim text so the answer is deterministic.""",

    3: """\
**Type 3 – Precise Mathematical Computation**

The corpus contains concrete numeric values (measurements, statistics, prices, counts, dates, percentages). These numbers become the parameters of the calculation problem.

Fusion strategy:
- Extract at least 2 specific numeric values from the corpus and use them as the problem's given quantities.
- Frame the calculation within the corpus's domain (e.g., "According to the report, trade volume is $5 trillion and the sea area is 3.5 million km^2...").
- The narrative context (who, where, why) must come from the corpus -- the numbers must also come from the corpus, not be invented.
- The `input` field must list the extracted numeric values in the exact format expected by the original code.""",

    4: """\
**Type 4 – Data Processing**

The corpus contains a structured list, table, or enumeration of entities with multiple attributes. This structured data becomes the dataset for the problem.

Fusion strategy:
- Identify the structured entries in the corpus (e.g., a list of regulations, a table of measurements, a ranked list of countries).
- Frame the problem as a data analysis task: filtering, sorting, grouping, or aggregating those entries.
- Present the data naturally in the exam question (as a table or bulleted list) within the corpus's narrative context.
- The `input` field must encode the structured data in the exact format expected by the original code (e.g., number of entries followed by attribute rows).""",

    5: """\
**Type 5 – State/Rule Simulation**

The corpus describes a step-by-step process, iterative procedure, lifecycle, or rule-based system. These rules become the simulation to execute.

Fusion strategy:
- Map the corpus's described process onto the algorithmic state transitions. Use the corpus's own terminology for states, transitions, and stopping conditions.
- Frame the problem as "given this process and an initial situation, what is the outcome after N steps / when does the system reach state X?"
- The initial state and any parameters should be derived from numeric or categorical values in the corpus where possible.
- The `input` field must encode the initial state and parameters in the exact format expected by the original code.""",
}

# fallback type label when type_level1 is missing or unrecognized
DEFAULT_TYPE = 3


# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# Corpus loading: bucket by suitable_types
# ──────────────────────────────────────────────────────────────

def load_corpus_by_type(corpus_path: str, max_corpus_len: int) -> dict:
    """
    Read classified JSONL files (can be a file or directory),
    return a {type_id: [text, ...]} dict where type_id is in {1,2,3,4,5}.
    A corpus entry with suitable_types=[2,3] goes into both bucket 2 and bucket 3.
    """
    buckets = defaultdict(list)

    if os.path.isfile(corpus_path):
        files = [corpus_path]
    else:
        files = sorted([
            os.path.join(corpus_path, f)
            for f in os.listdir(corpus_path)
            if f.endswith('.jsonl')
        ])

    total = 0
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = record.get('text', '').strip()
                suitable_types = record.get('suitable_types', [])
                if not text or not suitable_types:
                    continue
                text = text[:max_corpus_len]
                for t in suitable_types:
                    try:
                        tid = int(t)
                        if 1 <= tid <= 5:
                            buckets[tid].append(text)
                    except (ValueError, TypeError):
                        pass
                total += 1

    print(f"Loaded {total} corpus records into type buckets:")
    for tid in sorted(buckets):
        print(f"  Type {tid}: {len(buckets[tid])} entries")
    return dict(buckets)


# ──────────────────────────────────────────────────────────────
# QA loading
# ──────────────────────────────────────────────────────────────

def load_simplified_qa(input_path: str) -> list:
    data_list = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading simplified_qa"):
            line = line.strip()
            if not line:
                continue
            data_list.append(json.loads(line))
    print(f"Loaded {len(data_list)} QA samples")
    return data_list


# ──────────────────────────────────────────────────────────────
# Pairing: match by type
# ──────────────────────────────────────────────────────────────

def create_typed_pairs(
    qa_data: list,
    buckets: dict,
    num_samples_per_qa: int = 1,
    fallback_allow: bool = True,
) -> Dataset:
    """
    For each QA, sample num_samples_per_qa different corpus entries from the same-type bucket
    based on type_level1, expanding into num_samples_per_qa rows (each with a sample_idx field).
    fallback_allow=True: if the type bucket is empty, randomly sample from all corpus entries.
    fallback_allow=False: skip QA items with no matching corpus entries.
    """
    all_texts = [t for texts in buckets.values() for t in texts]

    pairs = []
    skipped = 0
    type_counter = defaultdict(int)

    for i, qa in enumerate(tqdm(qa_data, desc="Pairing QA with corpus")):
        raw_type = qa.get('type_level1', '')
        try:
            type_id = int(raw_type)
            if not (1 <= type_id <= 5):
                raise ValueError
        except (ValueError, TypeError):
            type_id = DEFAULT_TYPE

        pool = buckets.get(type_id, [])
        if not pool:
            if fallback_allow and all_texts:
                pool = all_texts
                type_counter['fallback'] += 1
            else:
                skipped += 1
                continue
        else:
            type_counter[type_id] += 1

        # Avoid duplicates: sample without replacement when k <= pool size, cycle when k > pool size
        if num_samples_per_qa <= len(pool):
            sampled_texts = random.sample(pool, num_samples_per_qa)
        else:
            sampled_texts = []
            while len(sampled_texts) < num_samples_per_qa:
                take = min(num_samples_per_qa - len(sampled_texts), len(pool))
                sampled_texts.extend(random.sample(pool, take))

        problem_text = qa.get('refined_question', '') or qa.get('question', '')
        # Concatenate full problem info so the model can see input/output format and examples
        full_problem_parts = [problem_text]
        if qa.get('input_format'):
            full_problem_parts.append(f"\nInput format:\n{qa['input_format']}")
        if qa.get('output_format'):
            full_problem_parts.append(f"\nOutput format:\n{qa['output_format']}")
        examples = qa.get('examples', [])
        if examples:
            full_problem_parts.append("\nExamples:")
            for idx_e, ex in enumerate(examples):
                full_problem_parts.append(
                    f"  Example {idx_e+1}:\n"
                    f"    Input:  {ex.get('input', '').strip()}\n"
                    f"    Output: {ex.get('output', '').strip()}"
                )
        full_problem = '\n'.join(full_problem_parts)

        base = {
            'id': qa.get('id', f'sample_{i}'),
            'contest_id': qa.get('contest_id', ''),
            'index': qa.get('index', ''),
            'title': qa.get('title', ''),
            'type_id': type_id,
            'original_problem': full_problem,
            'type_level1': qa.get('type_level1', ''),
            'type_level2': qa.get('type_level2', ''),
        }
        for k_idx, corpus_text in enumerate(sampled_texts):
            pairs.append({**base, 'corpus_text': corpus_text, 'sample_idx': k_idx})

    print(f"Created {len(pairs)} pairs from {len(qa_data)} QA items "
          f"(k={num_samples_per_qa}, skipped={skipped})")
    print(f"  Type distribution: { {k: v for k, v in sorted(type_counter.items())} }")
    return Dataset.from_list(pairs)


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────

def build_chat_texts(
    examples: dict,
    prompt_template: str,
    system_prompt: str,
    tokenizer,
    lang: str = 'en',
) -> list:
    instructions = TYPE_INSTRUCTIONS_ZH if lang == 'zh' else TYPE_INSTRUCTIONS
    chat_texts = []
    for corpus, problem, type_id in zip(
        examples['corpus_text'],
        examples['original_problem'],
        examples['type_id'],
    ):
        instruction = instructions.get(int(type_id), instructions[DEFAULT_TYPE])
        filled = (
            prompt_template
            .replace('{type_id}', str(type_id))
            .replace('{type_instruction}', instruction)
            .replace('{text}', corpus)
            .replace('{problem}', problem)
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
            batch, prompt_template, system_prompt, tokenizer, lang=args.lang
        )
        outputs = llm.generate(chat_texts, sampling_params)
        results.extend(out.outputs[0].text.strip() for out in outputs)

    return results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args):
    fix_seed(args.seed)

    # 1. Load classified corpus
    buckets = load_corpus_by_type(args.corpus_path, args.max_corpus_len)

    # 2. Load QA
    qa_data = load_simplified_qa(args.input_path)

    # 3. Pair QA with corpus
    dataset = create_typed_pairs(
        qa_data, buckets,
        num_samples_per_qa=args.num_samples_per_qa,
        fallback_allow=not args.strict_type,
    )
    if len(dataset) == 0:
        print("Error: no pairs created. Check corpus and QA types.")
        return

    # 4. Load prompt
    prompt_template = load_prompt(args.prompt_path)
    system_prompt = (
        load_prompt(args.system_prompt_path) if args.system_prompt_path
        else "You are a top-tier exam question designer specializing in realistic written assessments."
    )

    # 5. Load model
    print(f"Loading tokenizer from '{args.model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side='left',
        use_fast=args.use_fast,
    )

    if args.gpu_rank:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_rank))

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

    # 6. Sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )

    # 7. Sharding
    dataset = dataset.shard(args.shard[0], args.shard[1], contiguous=True)
    print(f"Using shard {args.shard[1]}/{args.shard[0]} ({len(dataset)} samples)")

    os.makedirs(args.output_path, exist_ok=True)
    total_samples = len(dataset)
    samples_per_shard = (total_samples + args.num_save_shards - 1) // args.num_save_shards

    start_time = time.time()

    # 8. Batch inference and save
    for shard_idx in range(args.num_save_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx   = min((shard_idx + 1) * samples_per_shard, total_samples)
        if start_idx >= end_idx:
            continue

        print(f"\nProcessing save-shard {shard_idx}/{args.num_save_shards} "
              f"(samples {start_idx}–{end_idx})...")
        shard_ds = dataset.select(range(start_idx, end_idx))

        responses = run_inference(
            args, tokenizer, llm,
            prompt_template, system_prompt, sampling_params, shard_ds,
        )

        shard_output_path = os.path.join(args.output_path, f"shard_{shard_idx}")
        inferenced = shard_ds.add_column(args.output_field, responses)
        print(f"Saving shard {shard_idx} to '{shard_output_path}'...")
        inferenced.save_to_disk(
            shard_output_path,
            num_proc=min(len(inferenced) - 1, args.save_workers),
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
        description="Type-aware context-based problem synthesis"
    )

    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--input_path',   type=str, required=True,
                        help="Path to simplified_qa.jsonl")
    parser.add_argument('--corpus_path',  type=str, required=True,
                        help="Classified corpus directory (JSONL files with suitable_types), "
                             "can be a single file or directory")
    parser.add_argument('--output_path',  type=str, required=True)
    parser.add_argument('--prompt_path',  type=str, required=True,
                        help="Path to synthesis_typed.txt")
    parser.add_argument('--system_prompt_path', type=str, default=None)
    parser.add_argument('--output_field', type=str, default='synthesis_result')

    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--max_tokens',    type=int,   default=2048)
    parser.add_argument('--max_model_len', type=int,   default=16384)
    parser.add_argument('--max_corpus_len',type=int,   default=4096)
    parser.add_argument('--temperature',   type=float, default=0.7)
    parser.add_argument('--seed',          type=int,   default=42)

    parser.add_argument('--shard',          type=int, nargs=2, default=[1, 0],
                        help="[num_shards, shard_index]")
    parser.add_argument('--num_save_shards',type=int, default=1)
    parser.add_argument('--gpu_rank',       type=int, nargs='+', default=None)
    parser.add_argument('--use_fast',       action='store_true')
    parser.add_argument('--save_workers',   type=int, default=4)
    parser.add_argument('--strict_type',    action='store_true',
                        help="Skip QA items with no matching corpus type instead of falling back to random pairing")
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'zh'],
                        help="Language for type fusion strategy descriptions (en/zh), must match the prompt file language")
    parser.add_argument('--num_samples_per_qa', type=int, default=1,
                        help="Number of different corpus entries to pair with each QA (k results); k>1 uses non-duplicate sampling when possible")

    args = parser.parse_args()
    print("args:", args)
    main(args)

