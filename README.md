# PaCoRe (Parallel Collaborative Reasoning) — PDC Project

This submission contains a clean, working PaCoRe-style prototype for a Parallel & Distributed Computing (PDC) project.

It includes two core pieces:

1) **Consensus engine**: aggregate multiple “worker” responses using several consensus algorithms.
2) **PaCoRe inference pipeline**: a branch → compact → synthesize loop for a single Hugging Face causal LM (e.g., Llama‑3B).

## What’s implemented

- **Consensus algorithms** in `src/consensus.py`
    - `majority_voting`
    - `weighted_voting`
    - `ensemble_average`
    - `ranking_based`
    - `borda_count`

- **PaCoRe inference** in `src/pacore_pipeline.py` + prompts in `src/pacore_prompts.py`
    - Multiple branches (parallel conceptual “workers”)
    - Compaction step
    - Synthesis step producing a tagged `FINAL_ANSWER:`

- **Optional PPO trainer skeleton** in `src/pacore_trainer.py` (uses TRL + a tiny JSONL math dataset)

## Setup (Windows)

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run demos

### 1) Consensus demo (no LLM required)

```bash
python examples\simple_usage.py
```

### 2) PaCoRe inference demo (LLM required)

```bash
python examples\run_pacore_pipeline.py "Solve: 23*17" --model meta-llama/Llama-3.2-3B-Instruct --branches 1 --branch_tokens 64 --compact_tokens 32 --synth_tokens 64
```

## Hugging Face access (Llama models)

Some Meta Llama repositories are gated on the Hugging Face Hub.

- Ensure you have access approved on the model page.
- Authenticate locally (recommended): `hf auth login`
- Or set an environment token: `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`

## Tests

```bash
python -m pytest -q
```

## Repository layout

```
PDC-Project/
    src/            # library code
    examples/       # runnable demos
    tests/          # unit tests
    data/           # tiny dataset used by trainer
    requirements.txt
    README.md
    FINAL_SUMMARY.txt
```
