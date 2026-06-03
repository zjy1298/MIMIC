#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpus type suitability filtering script.
Stage 1: Rule-based pre-filtering (lightweight feature detection, quickly eliminates clearly unsuitable texts).
Stage 2: LLM precise classification (vLLM batch inference, strictly determines suitable problem types 1-5).

Sharding mode:
  --shard_total N  --shard_start I  --shard_end J
  Single-pass file traversal, routes lines where line_num % shard_total is in [I, J] to the
  corresponding output_dir/shard_{k}.jsonl. Suitable for multi-process parallel processing of the same file.
Supports checkpoint resumption, checkpoint stored at output_dir/.ckpt_{shard_start}_{shard_end}
"""

import os
import re
import json
import time
import argparse

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ──────────────────────────────────────────────────────────────
# Stage 1: Rule-based pre-filtering
# ──────────────────────────────────────────────────────────────

_TEMPORAL_WORDS = [
    'first', 'then', 'next', 'after', 'step', 'phase', 'each round',
    'iteration', 'finally', 'subsequently', 'cycle', 'transition',
    'evolve', 'simulate', 'state', 'repeat', 'sequence', 'procedure',
    'until', 'while', 'at each', 'per step',
]
_SPATIAL_WORDS = [
    'north', 'south', 'east', 'west', 'left', 'right', 'grid', ' row',
    'column', 'adjacent', 'diagonal', 'coordinate', 'positioned',
    'map', 'layout', 'horizontal', 'vertical', 'above', 'below',
    'cell', 'matrix', 'floor plan', 'corner',
]
_MATH_DOMAINS = {'technology', 'education', 'economics', 'medical', 'natural_science'}


def rule_based_hints(text: str, domain_txt: str = '') -> list:
    """Lightweight feature detection. Returns a list of potentially suitable types (errs on the side of inclusion; LLM makes the final decision)."""
    if len(text) < 200:
        return []

    lower = text.lower()
    tokens = lower.split()
    n = max(len(tokens), 1)
    lines = text.split('\n')

    numeric_ratio = sum(1 for t in tokens if re.search(r'\d', t)) / n
    has_pipe_table = bool(re.search(r'\|.{2,}\|', text))
    list_lines = sum(
        1 for ln in lines
        if re.match(r'^\s*(\d+[\.\):]|[-*•])\s+\S', ln)
    )
    list_density = list_lines / max(len(lines), 1)
    temporal_score = sum(lower.count(w) for w in _TEMPORAL_WORDS)
    spatial_score = sum(lower.count(w) for w in _SPATIAL_WORDS)
    is_prose = (
        len(text) >= 400
        and not has_pipe_table
        and list_density < 0.15
        and numeric_ratio < 0.25
    )

    hints = set()
    if spatial_score >= 4:
        hints.add(1)
    if is_prose:
        hints.add(2)
    if numeric_ratio >= 0.07 or domain_txt in _MATH_DOMAINS:
        hints.add(3)
    if has_pipe_table or list_density >= 0.20:
        hints.add(4)
    if temporal_score >= 4:
        hints.add(5)

    return sorted(hints)


# ──────────────────────────────────────────────────────────────
# Checkpoint resumption
# ──────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str) -> int:
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r') as f:
            return json.load(f).get('processed_lines', 0)
    return 0


def save_checkpoint(ckpt_path: str, processed_lines: int):
    with open(ckpt_path, 'w') as f:
        json.dump({'processed_lines': processed_lines}, f)


# ──────────────────────────────────────────────────────────────
# Input stream
# ──────────────────────────────────────────────────────────────

def iter_records(path: str, skip_lines: int = 0):
    """
    Stream-read records, yield (line_num, text, domain_txt).
    line_num: 0-based global data line number (excluding blank lines).
    skip_lines: for checkpoint resumption, skip the first N lines.
    """
    count = 0
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            if count < skip_lines:
                count += 1
                continue
            try:
                row = json.loads(raw)
                text = row.get('text', '').strip()
                domain_txt = row.get('quality_domain_txt', '')
            except json.JSONDecodeError:
                text, domain_txt = '', ''
            yield count, text, domain_txt
            count += 1


# ──────────────────────────────────────────────────────────────
# LLM batch classification
# ──────────────────────────────────────────────────────────────

def build_chat_inputs(
    texts: list,
    hints_list: list,
    prompt_template: str,
    tokenizer,
    max_text_len: int,
) -> list:
    inputs = []
    for text, hints in zip(texts, hints_list):
        hints_str = ', '.join(map(str, hints)) if hints else 'none'
        filled = (
            prompt_template
            .replace('{text}', text[:max_text_len])
            .replace('{rule_hints}', hints_str)
        )
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": filled}],
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        inputs.append(chat)
    return inputs


def parse_llm_output(raw: str) -> tuple:
    """Returns (result_dict, parse_ok). parse_ok=False means JSON parsing completely failed."""
    raw = raw.strip()
    try:
        return json.loads(raw), True
    except json.JSONDecodeError:
        pass
    # Search backwards for the last valid JSON block (JSON is usually after thinking)
    for m in reversed(list(re.finditer(r'\{[^{}]*\}', raw, re.DOTALL))):
        try:
            return json.loads(m.group()), True
        except json.JSONDecodeError:
            continue
    return {'suitable_types': [], 'reasons': {}}, False


def classify_batch(
    texts: list,
    hints_list: list,
    llm,
    tokenizer,
    prompt_template: str,
    sampling_params,
    max_text_len: int,
) -> list:
    chat_inputs = build_chat_inputs(texts, hints_list, prompt_template, tokenizer, max_text_len)
    outputs = llm.generate(chat_inputs, sampling_params)
    results = []
    for out in outputs:
        raw_text = out.outputs[0].text
        parsed, parse_ok = parse_llm_output(raw_text)
        suitable = sorted({int(t) for t in parsed.get('suitable_types', [])
                           if str(t).isdigit() and 1 <= int(t) <= 5})
        results.append({
            'suitable_types': suitable,
            'reasons': parsed.get('reasons', {}),
            '_parse_ok': parse_ok,
            '_raw': raw_text,
        })
    return results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    rule_dir = os.path.join(args.output_dir, 'rule')
    llm_dir  = os.path.join(args.output_dir, 'llm')
    os.makedirs(rule_dir, exist_ok=True)
    os.makedirs(llm_dir,  exist_ok=True)

    shard_start = args.shard_start
    shard_end = args.shard_end
    shard_total = args.shard_total
    my_shards = set(range(shard_start, shard_end + 1))

    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    ckpt_path = os.path.join(args.output_dir, f'.ckpt_{prefix}{shard_start}_{shard_end}')
    start_line = load_checkpoint(ckpt_path)
    if start_line > 0:
        print(f"Resuming from global line {start_line} "
              f"(shards {shard_start}-{shard_end} of {shard_total})")

    with open(args.prompt_path, 'r', encoding='utf-8') as f:
        prompt_template = f.read()

    print(f"Loading tokenizer from {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side='left',
        use_fast=args.use_fast,
    )

    if args.gpu_rank:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu_rank))

    print("Loading model ...")
    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.model_name_or_path,
        dtype='auto',
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=True,
        max_model_len=args.max_model_len,
        # language_model_only=True,                                               
        # enable_expert_parallel=True,                                            
        additional_config={"gdn_prefill_backend": "triton"},
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=512,
        seed=args.seed,
    )

    # Each shard has two output files: rule (rule-based pre-filter) and llm (LLM precise classification)
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    rule_files = {
        s: open(os.path.join(rule_dir, f'{prefix}shard_{s}.jsonl'), 'a', encoding='utf-8')
        for s in my_shards
    }
    llm_files = {
        s: open(os.path.join(llm_dir, f'{prefix}shard_{s}.jsonl'), 'a', encoding='utf-8')
        for s in my_shards
    }

    # Global buffer: flush to LLM when batch_size is reached
    # Each record: (shard_idx, line_num, text, hints)
    buffer: list = []
    last_flushed_line = start_line - 1
    rule_written = 0
    llm_written = 0
    rule_skipped = 0
    t0 = time.time()

    def flush_buffer():
        nonlocal llm_written, last_flushed_line
        if not buffer:
            return
        texts  = [b[2] for b in buffer]
        hints  = [b[3] for b in buffer]
        results = classify_batch(texts, hints, llm, tokenizer,
                                 prompt_template, sampling_params, args.max_text_len)
        for (shard_idx, line_num, text, _hints), result in zip(buffer, results):
            parse_ok = result['_parse_ok']
            # if result['suitable_types'] or not parse_ok:
            record = {
                'text': text,
                'rule_hints': _hints,
                'suitable_types': result['suitable_types'],
                'reasons': result['reasons'],
            }
            # if not parse_ok:
            record['raw_output'] = result['_raw']
            llm_files[shard_idx].write(json.dumps(record, ensure_ascii=False) + '\n')
            llm_written += 1
        for f in rule_files.values():
            f.flush()
        for f in llm_files.values():
            f.flush()
        last_flushed_line = buffer[-1][1]
        save_checkpoint(ckpt_path, last_flushed_line + 1)
        buffer.clear()

    pbar = tqdm(desc=f"shards {shard_start}-{shard_end}", unit="lines")
    try:
        for line_num, text, domain_txt in iter_records(args.input_path, skip_lines=start_line):
            pbar.update(1)

            shard_idx = line_num % shard_total
            if shard_idx not in my_shards:
                continue
            if not text:
                continue

            hints = rule_based_hints(text, domain_txt)
            if not hints:
                rule_skipped += 1
                continue

            buffer.append((shard_idx, line_num, text[:args.max_text_len], hints))
            rule_files[shard_idx].write(
                json.dumps({'text': text[:args.max_text_len], 'rule_hints': hints},
                           ensure_ascii=False) + '\n'
            )
            rule_written += 1

            if len(buffer) >= args.batch_size:
                flush_buffer()
                pbar.set_postfix(
                    rule=rule_written,
                    llm=llm_written,
                    skip=rule_skipped,
                    elapsed=f"{time.time()-t0:.0f}s",
                )

        flush_buffer()  # Flush remaining data that doesn't fill a full batch
    finally:
        for f in rule_files.values():
            f.close()
        for f in llm_files.values():
            f.close()
        pbar.close()

    elapsed = time.time() - t0
    print(f"\nDone. rule-skipped={rule_skipped}, rule-written={rule_written}, "
          f"llm-written={llm_written}, elapsed={elapsed:.1f}s")
    print(f"Rule output : {rule_dir}")
    print(f"LLM  output : {llm_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Two-stage corpus type classifier with shard-parallel support"
    )

    # Model
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--use_fast', action='store_true')
    parser.add_argument('--gpu_rank', type=int, nargs='+', default=None)
    parser.add_argument('--max_model_len', type=int, default=4096)

    # Data
    parser.add_argument('--input_path', type=str, required=True,
                        help="Input file (each line is JSON with a 'text' field)")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory, shards written as shard_{N}.jsonl")
    parser.add_argument('--prompt_path', type=str, required=True)

    parser.add_argument('--output_prefix', type=str, default='',
                        help="Output filename prefix, files named as {prefix}_shard_{N}.jsonl")
    parser.add_argument('--shard_total', type=int, default=1,
                        help="Total number of shards (routed by line_num %% shard_total)")
    parser.add_argument('--shard_start', type=int, default=0,
                        help="Starting shard index for this process (inclusive)")
    parser.add_argument('--shard_end', type=int, default=0,
                        help="Ending shard index for this process (inclusive)")

    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_text_len', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Default shard_end = shard_total - 1 (covers all when not sharding)
    if args.shard_end == 0 and args.shard_total > 1:
        args.shard_end = args.shard_total - 1

    main(args)

