"""
Run a PaCoRe-style inference pass with Llama-3.2-3B (or any HF causal LM).
This showcases branch generation -> compaction -> synthesis on a toy problem.
"""
import argparse
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load a local .env if present (git-ignored). Do not override existing env vars.
# Preferred auth flow for gated models: run `hf auth login` so Transformers can
# use the saved token automatically.
load_dotenv(dotenv_path=ROOT / ".env", override=False)

# Debug hint (does not print the token): verifies `.env` injection worked.
_has_env_token = bool(os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"))
logger.info("HF token present in env: {}", _has_env_token)

from src.pacore_pipeline import PaCoRePipeline, PaCoRePipelineConfig
from src.pacore_prompts import PaCoRePromptConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaCoRe inference")
    parser.add_argument("problem", nargs="?", default="Prove that the sum of two even numbers is even.")
    parser.add_argument("--model", dest="model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument(
        "--device",
        default="auto",
        help="Where to run the model: auto|cpu|cuda|mps (auto picks best available).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading model code from the Hub (needed for some architectures).",
    )
    # Defaults chosen to complete in a reasonable time on CPU.
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--branch_tokens", type=int, default=96)
    parser.add_argument("--compact_tokens", type=int, default=64)
    parser.add_argument("--synth_tokens", type=int, default=96)
    return parser.parse_args()


def main():
    args = parse_args()
    prompt_cfg = PaCoRePromptConfig(
        branches=args.branches,
        branch_tokens=args.branch_tokens,
        compact_tokens=args.compact_tokens,
        synthesis_tokens=args.synth_tokens,
        rounds=args.rounds,
    )
    pipe_cfg = PaCoRePipelineConfig(
        model_name=args.model,
        prompt=prompt_cfg,
        trust_remote_code=bool(args.trust_remote_code),
        device=args.device,
    )
    pipeline = PaCoRePipeline(pipe_cfg)

    result = pipeline.run(args.problem)

    print("=" * 80)
    print("PaCoRe Inference Demo")
    print("=" * 80)
    print(f"Problem: {args.problem}\n")

    for i, (trace, note) in enumerate(zip(result["branches"], result["compact_notes"]), 1):
        print(f"--- Branch {i} ---")
        print(trace[:600])
        print("\nCompact note:")
        print(note)
        print("-" * 40)

    print("SYNTHESIS OUTPUT:\n", result["synthesis"])
    print("\nFINAL ANSWER:", result["final_answer"])


if __name__ == "__main__":
    main()
