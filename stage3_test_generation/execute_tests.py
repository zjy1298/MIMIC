#!/usr/bin/env python3
"""
execute_synthesized_inputs.py
Extracts synthesized problem inputs from synthesized_refined_v2,
executes the corresponding AC code via sandbox API to obtain golden outputs,
and outputs in standard eval_data JSONL format.

Usage:
  python execute_synthesized_inputs.py \
      --synthesized_dir ./data/input.jsonl \
      --qa_file ./data/input.jsonl \
      --output_file ./data/input.jsonl \
      --shard_start 0 --shard_end 79 \
      --concurrency 100
"""

import argparse
import asyncio
import json
import logging
import os
import random
import time
import glob

import aiohttp
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("exec_synth")

DEFAULT_SANDBOX_URL = os.environ.get(
    "SANDBOX_URL",
    "http://localhost:8080/sandbox",
)


async def call_sandbox(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    code: str,
    language: str,
    stdin: str = "",
    run_timeout: int = 30,
    request_timeout: int = 60,
    max_retries: int = 3,
    sandbox_url: str = DEFAULT_SANDBOX_URL,
) -> dict:
    url = f"{sandbox_url.rstrip('/')}/run_code"
    payload = {"code": code, "language": language, "run_timeout": run_timeout}
    if stdin:
        payload["stdin"] = stdin
    timeout = aiohttp.ClientTimeout(total=request_timeout)

    for attempt in range(1, max_retries + 1):
        try:
            async with semaphore:
                async with session.post(url, json=payload, timeout=timeout) as resp:
                    if resp.status == 200:
                        raw = await resp.read()
                        encoding = resp.headers.get("Content-Encoding", "")
                        if encoding == "br":
                            import brotli
                            raw = brotli.decompress(raw)
                        elif encoding == "gzip":
                            import gzip
                            raw = gzip.decompress(raw)
                        body = json.loads(raw)
                        run_result = body.get("run_result") or {}
                        return {
                            "success": run_result.get("return_code", -1) == 0,
                            "stdout": run_result.get("stdout", ""),
                            "stderr": run_result.get("stderr", ""),
                            "return_code": run_result.get("return_code", -1),
                        }
                    logger.warning("HTTP %d (attempt %d/%d)", resp.status, attempt, max_retries)
        except asyncio.TimeoutError:
            logger.warning("Timeout (attempt %d/%d)", attempt, max_retries)
        except aiohttp.ClientError as e:
            logger.warning("Error (attempt %d/%d): %s", attempt, max_retries, e)

        if attempt < max_retries:
            await asyncio.sleep(3.0 + random.uniform(0, 2))

    return {"success": False, "stdout": "", "stderr": "max retries exceeded", "return_code": -1}


def load_ac_codes(qa_file: str) -> dict:
    """Load AC code for each problem ID. Returns {pid: (code, language)}."""
    ac_codes = {}
    with open(qa_file, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            pid = d.get('id', '')
            if pid in ac_codes:
                continue
            for sol in (d.get('accepted_solutions') or []):
                code = sol.get('source', '')
                lang = sol.get('programmingLanguage', '')
                if code:
                    ac_codes[pid] = (code, lang)
                    break
    logger.info(f"Loaded AC codes for {len(ac_codes)} problems")
    return ac_codes


def normalize_language(lang: str) -> str:
    """Normalize language name for sandbox API."""
    lang_lower = lang.lower()
    if 'python' in lang_lower:
        return 'python3'
    elif 'java' in lang_lower:
        return 'java'
    elif 'c++' in lang_lower or 'gnu c++' in lang_lower or 'g++' in lang_lower:
        return 'cpp'
    elif lang_lower in ('c', 'gnu c'):
        return 'c'
    elif 'rust' in lang_lower:
        return 'rust'
    elif 'go' in lang_lower:
        return 'go'
    elif 'kotlin' in lang_lower:
        return 'kotlin'
    else:
        return lang_lower


def load_synthesized_records(synthesized_dir: str, shard_start: int, shard_end: int) -> list:
    """Load synthesized records from specified shard range."""
    records = []
    for shard_idx in range(shard_start, shard_end + 1):
        shard_dir = os.path.join(synthesized_dir, f"shard_{shard_idx}")
        inner = os.path.join(shard_dir, "shard_0")
        path = inner if os.path.isdir(inner) else shard_dir
        if not os.path.isdir(path):
            continue
        try:
            ds = load_from_disk(path)
            for r in ds:
                # Parse refined_result first, fallback to synthesis_result
                problem_text = ''
                input_text = ''
                for field in ['refined_result', 'synthesis_result']:
                    try:
                        obj = json.loads(r[field])
                        p = obj.get('problem', '')
                        i = obj.get('input', '')
                        if p and i is not None:
                            problem_text = p
                            input_text = i
                            break
                    except (json.JSONDecodeError, TypeError, KeyError):
                        continue

                if not problem_text or not input_text.strip():
                    continue

                records.append({
                    'id': r['id'],
                    'title': r['title'],
                    'type_id': r['type_id'],
                    'sample_idx': r['sample_idx'],
                    'problem': problem_text,
                    'input': input_text,
                    'rating': 0,
                })
        except Exception as e:
            logger.warning(f"Skip shard_{shard_idx}: {e}")

    logger.info(f"Loaded {len(records)} synthesized records from shards {shard_start}-{shard_end}")
    return records


def load_done_keys(output_file: str) -> set:
    """Load already-processed keys for checkpoint resume."""
    done = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(d.get('eval_id', ''))
                except:
                    pass
    return done


async def execute_batch(records, ac_codes, args, done_keys, output_file):
    """Execute all records through sandbox."""
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2, ttl_dns_cache=300)
    session_kwargs = {"connector": connector, "auto_decompress": False}

    write_lock = asyncio.Lock()
    pbar = tqdm(total=len(records), desc="Executing")
    stats = {"success": 0, "fail": 0, "skip_no_code": 0, "skip_done": 0}

    async with aiohttp.ClientSession(**session_kwargs) as session:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def do_one(record):
            eval_id = f"{record['id']}::synth::{record['sample_idx']}"

            if eval_id in done_keys:
                stats["skip_done"] += 1
                pbar.update(1)
                return

            pid = record['id']
            if pid not in ac_codes:
                stats["skip_no_code"] += 1
                pbar.update(1)
                return

            code, lang = ac_codes[pid]
            lang_norm = normalize_language(lang)

            result = await call_sandbox(
                session, semaphore,
                code=code,
                language=lang_norm,
                stdin=record['input'],
                run_timeout=args.run_timeout,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                sandbox_url=args.sandbox_url,
            )

            if result['success'] and result['stdout'].strip():
                output_record = {
                    "eval_id": eval_id,
                    "problem_id": pid,
                    "title": record['title'],
                    "source": "synthesized",
                    "type_id": record['type_id'],
                    "problem": record['problem'],
                    "input": record['input'],
                    "expected_output": result['stdout'].strip(),
                    "rating": record.get('rating', 0),
                }
                async with write_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                stats["success"] += 1
            else:
                stats["fail"] += 1

            pbar.update(1)

        tasks = [do_one(r) for r in records]
        await asyncio.gather(*tasks)

    pbar.close()
    return stats


async def amain():
    parser = argparse.ArgumentParser(description="Execute synthesized inputs via sandbox")
    parser.add_argument("--synthesized_dir", type=str, required=True)
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--shard_start", type=int, default=0)
    parser.add_argument("--shard_end", type=int, default=79)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--run_timeout", type=int, default=30)
    parser.add_argument("--request_timeout", type=int, default=60)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sandbox_url", type=str, default=DEFAULT_SANDBOX_URL)
    args = parser.parse_args()

    ac_codes = load_ac_codes(args.qa_file)
    records = load_synthesized_records(args.synthesized_dir, args.shard_start, args.shard_end)

    # Add rating from QA
    ratings = {}
    with open(args.qa_file, 'r') as f:
        for line in f:
            d = json.loads(line)
            ratings[d.get('id', '')] = d.get('rating', 0) or 0
    for r in records:
        r['rating'] = ratings.get(r['id'], 0)

    done_keys = load_done_keys(args.output_file)
    logger.info(f"Checkpoint: {len(done_keys)} already done")

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)

    stats = await execute_batch(records, ac_codes, args, done_keys, args.output_file)
    logger.info(f"Done. success={stats['success']}, fail={stats['fail']}, "
                f"skip_no_code={stats['skip_no_code']}, skip_done={stats['skip_done']}")


if __name__ == "__main__":
    asyncio.run(amain())
