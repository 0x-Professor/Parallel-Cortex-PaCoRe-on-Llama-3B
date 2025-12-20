# PaCoRe Implementation Guide

## Quick Start Guide

### Step 1: Setup Project Structure

Run the setup script to create necessary directories:

```bash
# On Windows
setup.bat

# On Linux/Mac
python setup_project.py
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Test Installation

Run the consensus example to verify setup:

```bash
python examples/simple_usage.py
```

### Step 4: Run Tests

```bash
pytest tests/ -v
```

## Architecture Overview

### 1. Master Node (`src/master.py`)
- Coordinates task distribution
- Aggregates responses from workers
- Implements REST API for task submission
- Manages worker pool

**Key Features:**
- Parallel task execution using Ray or asyncio
- Multiple consensus strategies
- Load balancing across workers
- Health monitoring

### 2. Worker Node (`src/worker.py`)
- Loads and runs language models
- Processes individual prompts
- Returns responses with confidence scores
- Supports multiple model backends

**Supported Models:**
- microsoft/phi-2 (2.7B parameters)
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B parameters)
- facebook/opt-1.3b (1.3B parameters)

### 3. Consensus Engine (`src/consensus.py`)
- Aggregates multiple model responses
- Implements 5 consensus algorithms:
  1. **Majority Voting**: Most common response wins
  2. **Weighted Voting**: Confidence-weighted selection
  3. **Ensemble Average**: Combines multiple responses
  4. **Ranking Based**: Multi-factor scoring
  5. **Borda Count**: Preference-based voting

## Usage Examples

### Example 1: Basic Consensus

```python
from consensus import ConsensusEngine

# Simulate model responses
responses = [
    {"response": "Answer A", "confidence": 0.9},
    {"response": "Answer B", "confidence": 0.7},
    {"response": "Answer A", "confidence": 0.85}
]

# Apply consensus
engine = ConsensusEngine(strategy="weighted_voting")
result = engine.aggregate(responses)

print(f"Consensus: {result['text']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Example 2: Start Master Node

```bash
python src/master.py --config config.yaml
```

Then send requests via HTTP:

```python
import requests

response = requests.post('http://localhost:5000/process', json={
    "prompt": "What is parallel computing?",
    "task_id": "task_001"
})

print(response.json())
```

### Example 3: Test Worker Node

```bash
python src/worker.py --test --config config.yaml
```

## Configuration

Edit `config.yaml` to customize:

- **Models**: Add/remove models from the pool
- **Consensus**: Change strategy and thresholds
- **Parallel Backend**: Choose between Ray, MPI, or gRPC
- **Resources**: Set batch size, max_length, device

## Performance Optimization

### 1. GPU Acceleration
```yaml
worker:
  device: "cuda"  # Use GPU if available
```

### 2. Batch Processing
```yaml
worker:
  batch_size: 16  # Process multiple prompts together
```

### 3. Model Selection
Use smaller, faster models for quicker responses:
- TinyLlama (1.1B) - Fastest
- Phi-2 (2.7B) - Balanced
- OPT-1.3b (1.3B) - Good quality

### 4. Parallel Replicas
```yaml
parallel:
  num_replicas: 4  # More workers = faster processing
```

## Distributed Deployment

### Single Machine (Multiple Workers)

```bash
# Terminal 1: Master
python src/master.py

# Terminal 2: Worker 1
python src/worker.py --node-id 1

# Terminal 3: Worker 2  
python src/worker.py --node-id 2
```

### Multiple Machines (Ray Cluster)

```bash
# Head node
ray start --head --port=6379

# Worker nodes
ray start --address=<head-ip>:6379

# Run master
python src/master.py --config config.yaml
```

## Extending the Framework

### Add Custom Consensus Strategy

```python
# In src/consensus.py

def custom_strategy(self, responses):
    # Your logic here
    return {
        "text": "consensus result",
        "confidence": 0.85
    }

# Register strategy
self.strategies["custom"] = self.custom_strategy
```

### Add New Model

```yaml
# In config.yaml
models:
  pool:
    - name: "my-model"
      path: "org/model-name"
      weight: 1.0
```

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch_size or use CPU

```yaml
worker:
  batch_size: 4
  device: "cpu"
```

### Issue: Model download fails
**Solution**: Pre-download models

```python
from transformers import AutoModel, AutoTokenizer

model_name = "microsoft/phi-2"
AutoTokenizer.from_pretrained(model_name)
AutoModel.from_pretrained(model_name)
```

### Issue: Ray connection error
**Solution**: Use asyncio backend

```yaml
parallel:
  backend: "asyncio"
```

## Research Context

PaCoRe demonstrates that:
1. Multiple smaller models can match/exceed larger model performance
2. Consensus mechanisms improve answer quality
3. Parallel processing reduces latency
4. Distributed computing enables scalability

## Next Steps

1. **Benchmark**: Compare single large model vs PaCoRe ensemble
2. **Optimize**: Fine-tune consensus parameters for your use case
3. **Scale**: Deploy across multiple machines
4. **Experiment**: Try different model combinations

## References

- Original PaCoRe Framework
- Parallel Computing Principles
- Consensus Algorithms in Distributed Systems
- Language Model Ensembling

## Contributing

Feel free to:
- Add new consensus strategies
- Support additional models
- Improve parallel processing
- Optimize performance

## License

MIT License
