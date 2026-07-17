**[Updated] Response to W5: Limited evaluation scale**

We thank the reviewer for the supportive feedback. We have now completed the RL (GRPO) experiment on Qwen3-14B and updated the full results in the anonymous repository.

| | **ARC** | **BBH** | **C2R** | **GPQA** | **MMLU** | **DROP** | **Gen Avg** | **GSM8K** | **GSM+** | **MATH** | **AIME** | **Math Avg** | **HEval** | **CruxE** | **MBPP** | **Code Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | 83.5 | 81.9 | 42.4 | 46.4 | 72.7 | 55.5 | 63.7 | 91.4 | 76.1 | 82.2 | 30.0 | 69.9 | 33.4 | 73.5 | 42.4 | 49.8 |
| + SFT | 82.6 | **82.5** | 44.7 | 47.7 | 72.1 | **60.9** | 65.1 | **93.0** | 75.3 | **86.5** | 30.0 | 71.2 | **38.9** | 78.3 | **47.5** | **54.9** |
| + GRPO | **83.9** | 82.3 | **45.4** | **51.1** | **73.7** | 58.4 | **65.8** | 92.9 | **77.8** | 83.6 | **40.0** | **73.6** | 35.1 | **82.6** | 46.7 | 54.8 |

GRPO further improves upon Qwen3-14B, especially on math reasoning (Math Avg: 71.2 → 73.6, AIME: 30.0 → 40.0), confirming that RL leverages the model's inherent capabilities for continued gains. These results further consolidate our conclusion that MIMIC scales effectively beyond ≤8B models.
