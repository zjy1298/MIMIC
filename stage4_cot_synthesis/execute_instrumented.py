#!/usr/bin/env python3
"""
execute_instrumented.py
Executes all test inputs (official, generated, synthesized) using instrumented code,
and collects TRACE output via sandbox API for subsequent CoT synthesis.

Data sources:
  - instrumented_code_new: instrumented AC code
  - executed_tests: input+output of generated test cases (level A/B/C)
  - simplified_qa_enriched_v2: input+output of official tests
  - exec_synthesized: input+output of synthesized inputs

Usage:
  python execute_instrumented.py \
      --instrumented_dir ./data/input.jsonl \
      --executed_tests_dir ./data/input.jsonl \
      --qa_file ./data/input.jsonl \
      --synthesized_exec ./data/input.jsonl \
      --output_file ./data/input.jsonl \
      --concurrency 100
"""

import argparse
import asyncio
import json
import logging
import os
import random
import glob

import aiohttp
from tqdm import tqdm
from datasets import load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("exec_trace")

DEFAULT_SANDBOX_URL = os.environ.get(
    "SANDBOX_URL",
    "http://localhost:8080/sandbox",
)


async def call_sandbox(session, semaphore, code, language, stdin="",
                       run_timeout=30, request_timeout=60, max_retries=3,
                       sandbox_url=DEFAULT_SANDBOX_URL):
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
                        }
                    logger.warning("HTTP %d (attempt %d/%d)", resp.status, attempt, max_retries)
        except asyncio.TimeoutError:
            logger.warning("Timeout (attempt %d/%d)", attempt, max_retries)
        except aiohttp.ClientError as e:
            logger.warning("Error (attempt %d/%d): %s", attempt, max_retries, e)

        if attempt < max_retries:
            await asyncio.sleep(3.0 + random.uniform(0, 2))

    return {"success": False, "stdout": "", "stderr": "max retries exceeded"}


def parse_trace_output(stdout: str):
    """Separate TRACE lines from actual output."""
    traces = []
    output_lines = []
    for line in stdout.split('\n'):
        if line.strip().startswith('[TRACE]'):
            traces.append(line.strip())
        else:
            output_lines.append(line)
    return traces, '\n'.join(output_lines).strip()


def load_instrumented_codes(instrumented_dir: str) -> dict:
    """Load instrumented code per problem ID.
    Returns {pid: {code, language, analysis, trace_plan}}."""
    codes = {}
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
            if pid in codes:
                continue
            try:
                result = json.loads(r['instrument_result'])
                inst_code = result.get('instrumented_code', '')
                if inst_code:
                    codes[pid] = {
                        'code': inst_code,
                        'language': r['language'],
                        'analysis': result.get('analysis', ''),
                        'trace_plan': result.get('trace_plan', []),
                    }
            except (json.JSONDecodeError, TypeError):
                continue

    logger.info(f"Loaded instrumented code for {len(codes)} problems")
    return codes


def load_all_test_inputs(executed_tests_dir, qa_file, synthesized_exec_dir) -> list:
    """Collect all test inputs from three sources."""
    test_inputs = []

    # 1. Generated tests (level A/B/C)
    exec_files = sorted(glob.glob(os.path.join(executed_tests_dir, 'exec_shard_*.jsonl')))
    for fpath in exec_files:
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    r = json.loads(line)
                except:
                    continue
                pid = r.get('id', '')
                for tc in r.get('test_cases', []):
                    if tc.get('status') == 'success' and tc.get('input') and tc.get('output'):
                        test_inputs.append({
                            'problem_id': pid,
                            'source': f"level_{tc.get('level', 'C')}",
                            'input': tc['input'],
                            'expected_output': tc['output'],
                        })

    logger.info(f"Loaded {len(test_inputs)} generated test inputs")

    # 2. Official tests
    official_count = 0
    with open(qa_file, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
            except:
                continue
            pid = d.get('id', '')
            for t in (d.get('official_tests') or []):
                if t.get('input') and t.get('output'):
                    test_inputs.append({
                        'problem_id': pid,
                        'source': 'official',
                        'input': t['input'],
                        'expected_output': t['output'],
                    })
                    official_count += 1

    logger.info(f"Loaded {official_count} official test inputs")

    # 3. Synthesized inputs (already executed with golden output)
    synth_count = 0
    if synthesized_exec_dir and os.path.isdir(synthesized_exec_dir):
        synth_files = sorted(glob.glob(os.path.join(synthesized_exec_dir, 'exec_synth_*.jsonl')))
        for fpath in synth_files:
            with open(fpath, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except:
                        continue
                    test_inputs.append({
                        'problem_id': d.get('problem_id', ''),
                        'source': 'synthesized',
                        'input': d.get('input', ''),
                        'expected_output': d.get('expected_output', ''),
                    })
                    synth_count += 1

    logger.info(f"Loaded {synth_count} synthesized test inputs")
    logger.info(f"Total test inputs: {len(test_inputs)}")
    return test_inputs


def load_done_keys(output_file):
    done = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add(d.get('trace_id', ''))
                except:
                    pass
    return done


async def execute_batch(test_inputs, inst_codes, args, done_keys, output_file):
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2, ttl_dns_cache=300)
    session_kwargs = {"connector": connector, "auto_decompress": False}

    write_lock = asyncio.Lock()
    pbar = tqdm(total=len(test_inputs), desc="Executing traces")
    stats = {"success": 0, "fail": 0, "skip_no_code": 0, "skip_done": 0, "output_mismatch": 0}

    async with aiohttp.ClientSession(**session_kwargs) as session:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def do_one(idx, ti):
            pid = ti['problem_id']
            trace_id = f"{pid}::{ti['source']}::{idx}"

            if trace_id in done_keys:
                stats["skip_done"] += 1
                pbar.update(1)
                return

            if pid not in inst_codes:
                stats["skip_no_code"] += 1
                pbar.update(1)
                return

            ic = inst_codes[pid]
            result = await call_sandbox(
                session, semaphore,
                code=ic['code'],
                language=ic['language'],
                stdin=ti['input'],
                run_timeout=args.run_timeout,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                sandbox_url=args.sandbox_url,
            )

            if result['success'] and result['stdout']:
                traces, actual_output = parse_trace_output(result['stdout'])
                output_matched = actual_output.strip() == ti['expected_output'].strip()

                record = {
                    "trace_id": trace_id,
                    "problem_id": pid,
                    "source": ti['source'],
                    "input": ti['input'],
                    "expected_output": ti['expected_output'],
                    "actual_output": actual_output,
                    "output_matched": output_matched,
                    "traces": traces,
                    "raw_stdout": result['stdout'],
                    "analysis": ic['analysis'],
                }
                async with write_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')

                if output_matched:
                    stats["success"] += 1
                else:
                    stats["output_mismatch"] += 1
            else:
                stats["fail"] += 1

            pbar.update(1)

        tasks = [do_one(i, ti) for i, ti in enumerate(test_inputs)]
        await asyncio.gather(*tasks)

    pbar.close()
    return stats


async def amain():
    parser = argparse.ArgumentParser(description="Execute instrumented code with all test inputs")
    parser.add_argument("--instrumented_dir", type=str, required=True)
    parser.add_argument("--executed_tests_dir", type=str, required=True)
    parser.add_argument("--qa_file", type=str, required=True)
    parser.add_argument("--synthesized_exec", type=str, default="")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--run_timeout", type=int, default=30)
    parser.add_argument("--request_timeout", type=int, default=60)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sandbox_url", type=str, default=DEFAULT_SANDBOX_URL)
    parser.add_argument("--shard", type=int, nargs=2, default=[1, 0],
                        help="[num_shards, shard_index] to split test inputs across jobs")
    args = parser.parse_args()

    inst_codes = load_instrumented_codes(args.instrumented_dir)
    test_inputs = load_all_test_inputs(
        args.executed_tests_dir, args.qa_file, args.synthesized_exec
    )

    # Shard the test inputs
    num_shards, shard_idx = args.shard
    total = len(test_inputs)
    per_shard = (total + num_shards - 1) // num_shards
    start = shard_idx * per_shard
    end = min(start + per_shard, total)
    test_inputs = test_inputs[start:end]
    logger.info(f"Shard {shard_idx}/{num_shards}: processing {len(test_inputs)} of {total} test inputs")

    done_keys = load_done_keys(args.output_file)
    logger.info(f"Checkpoint: {len(done_keys)} already done")

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)

    stats = await execute_batch(test_inputs, inst_codes, args, done_keys, args.output_file)
    logger.info(
        f"Done. success={stats['success']}, fail={stats['fail']}, "
        f"output_mismatch={stats['output_mismatch']}, "
        f"skip_no_code={stats['skip_no_code']}, skip_done={stats['skip_done']}"
    )


if __name__ == "__main__":
    asyncio.run(amain())
