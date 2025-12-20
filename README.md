# PaCoRe: Parallel Collaborative Reasoning Framework

## Overview
PaCoRe (Parallel Collaborative Reasoning) is a distributed framework that enables smaller language models to achieve performance comparable to larger models through collaborative reasoning and parallel processing.

## Key Features
- **Parallel Model Execution**: Run multiple smaller models in parallel
- **Collaborative Reasoning**: Aggregate responses from multiple models
- **Distributed Processing**: Leverage multiple compute nodes
- **Efficient Resource Usage**: Achieve better results with lower computational cost
- **PaCoRe RL Pipeline**: PaCoRe-style branch/compact/synthesize loop with PPO fine-tuning for Llama‑3B (see `src/pacore_pipeline.py` and `src/pacore_trainer.py`).

## Architecture
```
┌─────────────────────────────────────────────┐
│           Master Coordinator                │
│  - Task Distribution                        │
│  - Response Aggregation                     │
│  - Consensus Building                       │
└─────────────────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    │         │         │
┌───▼───┐ ┌──▼────┐ ┌──▼────┐
│Model 1│ │Model 2│ │Model 3│
│Worker │ │Worker │ │Worker │
└───────┘ └───────┘ └───────┘
```

## Components
1. **Master Node**: Coordinates tasks and aggregates results
2. **Worker Nodes**: Execute models in parallel
3. **Consensus Engine**: Combines outputs using voting/weighted strategies
4. **Communication Layer**: MPI/gRPC for inter-node communication

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Start master node
python src/master.py --config config.yaml

# Start worker nodes
python src/worker.py --node-id 1 --master-host localhost:5000

# Run PaCoRe inference demo (Llama-3.2-3B)
python examples/run_pacore_pipeline.py "Prove that the sum of two even numbers is even."

# Train PaCoRe with PPO on tiny math set
python -m src.pacore_trainer

Windows / Python 3.13 note: `ray` has no wheel for this combo. You can either
- skip `ray` (not required for the PaCoRe demo/trainer), or
- use WSL/conda with Python 3.10–3.11 to enable `ray` if you need distributed execution.
```

## Project Structure
```
PDC-Project/
├── src/
│   ├── master.py          # Master coordinator
│   ├── worker.py          # Worker node
│   ├── consensus.py       # Consensus algorithms
│   └── models/            # Model wrappers
├── config/
│   └── config.yaml        # Configuration
├── tests/
│   └── test_*.py          # Unit tests
└── requirements.txt       # Dependencies
```

## Research Paper Implementation
This implementation is based on the PaCoRe framework for collaborative reasoning.

## License
MIT
