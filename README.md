# MIMIC : Mining Intelligence from Machine-executed Instrumented Code

Official code and prompts for the paper: *The Imitation Game: When LLMs Learn to Reason Like Programs via Code-Centric Reasoning Data Synthesis*.

## Overview

MIMIC transforms raw competitive programming problems into verifiable natural-language reasoning training data through a four-stage pipeline:

1. **Problem Classification & Kernel Extraction** — Categorizes problems into 5 reasoning paradigms and isolates the mathematical kernel from contest-specific narrative.
2. **Problem Synthesis via Narrative Fusion** — Fuses mathematical kernels with real-world corpus text to produce story-wrapped exam questions.
3. **Code-Guided Test Synthesis & Input Fusion** — Generates diverse test inputs by analyzing code control flow, then weaves new inputs into story narratives.
4. **Trace-Grounded CoT Synthesis** — Instruments accepted code with diagnostic probes, collects execution traces, and synthesizes hallucination-free chain-of-thought reasoning.

## Repository Structure

```
├── prompts/                         # LLM prompt templates for each stage
├── stage1_classification/           # Problem classification & kernel extraction
│   ├── classify_problems.py         # Classify problems into 5 reasoning types
│   ├── extract_kernel.py            # Extract mathematical kernel from problems
│   └── classify_corpus.py           # Classify corpus text for type-aware sampling
├── stage2_synthesis/                # Story wrapping & narrative refinement
│   ├── synthesize_problems.py       # Type-aware narrative fusion
│   └── refine_synthesis.py          # Narrative smoothing pass
├── stage3_test_generation/          # Test input generation & fusion
│   ├── generate_tests.py            # Control-flow-guided test generation
│   ├── execute_tests.py             # Execute inputs via sandbox to get golden outputs
│   └── fuse_input.py                # Fuse new inputs into story narratives
├── stage4_cot_synthesis/            # Code instrumentation & CoT generation
│   ├── instrument_code.py           # Insert [TRACE] diagnostic probes
│   ├── execute_instrumented.py      # Collect execution traces via sandbox
│   └── synthesize_cot.py            # Generate trace-grounded CoT reasoning
├── data_construction/
│   └── build_sft_data.py            # Convert pipeline output to SFT training format
└── taxonomy.txt                     # 5-type reasoning paradigm taxonomy
```

## Requirements

- Python 3.10+
- vLLM (for LLM inference)
- aiohttp (for sandbox API calls)
- transformers

## Usage

Each stage script uses vLLM for batched LLM inference with shard-based parallelism. Example:

```bash
# Stage 1: Classify problems
python stage1_classification/classify_problems.py \
    --model_name_or_path <model_path> \
    --input_path ./data/problems.jsonl \
    --output_path ./output/classified.jsonl \
    --prompt_path ./prompts/classify.txt \
    --tensor_parallel_size 1

# Stage 2: Synthesize story-wrapped questions
python stage2_synthesis/synthesize_problems.py \
    --model_name_or_path <model_path> \
    --input_path ./data/simplified_qa.jsonl \
    --corpus_path ./data/corpus/ \
    --output_path ./output/synthesized.jsonl \
    --prompt_path ./prompts/synthesis_typed.txt

# Stage 4: Synthesize trace-grounded CoT
python stage4_cot_synthesis/synthesize_cot.py \
    --model_name_or_path <model_path> \
    --input_path ./data/traced_data.jsonl \
    --output_path ./output/cot.jsonl \
    --prompt_path ./prompts/synthesize_cot.txt
```

## Prompt Templates

All prompt templates are in the `prompts/` directory. Each template uses `{placeholder}` syntax for variable substitution. See individual files for detailed instructions.

## Citation

```bibtex
@article{anonymous2025mimic,
  title={The Imitation Game: When LLMs Learn to Reason Like Programs via Code-Centric Reasoning Data Synthesis},
  author={Anonymous},
  year={2025}
}
```

## License

This project is released under the MIT License.
