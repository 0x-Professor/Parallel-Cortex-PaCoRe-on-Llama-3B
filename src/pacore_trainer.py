"""
Minimal PPO-based trainer for PaCoRe using TRL.
This treats each stage (branch, compaction, synthesis) as an action with the
same terminal reward. The goal is to encourage the model to improve messages
that lead to a better final answer.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer
from loguru import logger

from .pacore_prompts import (
    PaCoRePromptConfig,
    build_branch_prompt,
    build_compaction_prompt,
    build_synthesis_prompt,
    parse_final_answer,
)


@dataclass
class RewardConfig:
    reward_on_missing: float = -0.2
    reward_on_match: float = 1.0
    reward_on_mismatch: float = 0.0
    tolerate_case: bool = True


@dataclass
class TrainerConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    prompt: PaCoRePromptConfig = field(default_factory=PaCoRePromptConfig)
    ppo: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            learning_rate=1e-6,
            batch_size=2,
            mini_batch_size=1,
            gradient_accumulation_steps=1,
        )
    )
    max_dataset: int = 32


class MathReward:
    """Simple exact-match reward for math or short-form answers."""

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def __call__(self, prediction: str, gold: str) -> float:
        if prediction is None:
            return self.cfg.reward_on_missing
        pred = prediction.strip()
        truth = gold.strip()
        if self.cfg.tolerate_case:
            pred, truth = pred.lower(), truth.lower()
        return self.cfg.reward_on_match if pred == truth else self.cfg.reward_on_mismatch


def load_jsonl_dataset(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
    return items


class PaCoReTrainer:
    def __init__(self, cfg: TrainerConfig, reward_fn: Callable[[str, str], float]):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.ppo_trainer = PPOTrainer(cfg.ppo, self.model, self.tokenizer)
        self.reward_fn = reward_fn

    @torch.no_grad()
    def _generate(self, prompts: List[str], max_new_tokens: int, temperature: float) -> List[str]:
        outputs: List[str] = []
        for prompt in prompts:
            query = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            generated = self.model.generate(
                **query,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=self.cfg.prompt.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            outputs.append(self.tokenizer.decode(generated[0], skip_special_tokens=True))
        return outputs

    def train_epoch(self, dataset: List[Dict[str, Any]]):
        prompt_cfg = self.cfg.prompt
        for example in dataset[: self.cfg.max_dataset]:
            problem = example["problem"]
            answer = example["answer"]

            branch_prompts = [build_branch_prompt(problem) for _ in range(prompt_cfg.branches)]
            branch_outputs = self._generate(branch_prompts, prompt_cfg.branch_tokens, prompt_cfg.temperature_branch)

            compact_prompts = [build_compaction_prompt(txt, prompt_cfg.compact_tokens) for txt in branch_outputs]
            compact_notes = self._generate(compact_prompts, prompt_cfg.compact_tokens, temperature=0.3)

            synth_prompt = build_synthesis_prompt(problem, compact_notes)
            synth_output = self._generate([synth_prompt], prompt_cfg.synthesis_tokens, prompt_cfg.temperature_synthesis)[0]
            final_answer = parse_final_answer(synth_output)
            reward = self.reward_fn(final_answer, answer)

            queries: List[torch.Tensor] = []
            responses: List[torch.Tensor] = []
            rewards: List[float] = []

            # Each stage gets the same terminal reward so PPO can credit assign.
            for q, r in [
                *zip(branch_prompts, branch_outputs),
                *zip(compact_prompts, compact_notes),
                (synth_prompt, synth_output),
            ]:
                queries.append(self.tokenizer(q, return_tensors="pt").input_ids[0])
                responses.append(self.tokenizer(r, return_tensors="pt").input_ids[0])
                rewards.append(reward)

            stats = self.ppo_trainer.step(queries, responses, rewards)
            logger.info(
                "PPO update | reward={:.2f} | kl={:.3f} | loss={:.3f}",
                reward,
                float(stats["objective/kl"].mean().item()),
                float(stats["ppo/loss"].mean().item()),
            )

    def save(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.ppo_trainer.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Saved trained model to {}", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaCoRe PPO trainer")
    parser.add_argument("--dataset", default="data/math_train.jsonl", help="Path to JSONL with problem/answer")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="HF model name or path")
    parser.add_argument("--output", default="models/pacore-3b", help="Where to save the trained model")
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--branch_tokens", type=int, default=160)
    parser.add_argument("--compact_tokens", type=int, default=80)
    parser.add_argument("--synth_tokens", type=int, default=160)
    parser.add_argument("--temp_branch", type=float, default=0.7)
    parser.add_argument("--temp_synth", type=float, default=0.35)
    args = parser.parse_args()

    data_path = Path(args.dataset)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}. Provide JSONL with problem/answer pairs.")

    dataset = load_jsonl_dataset(data_path)
    prompt_cfg = PaCoRePromptConfig(
        branches=args.branches,
        branch_tokens=args.branch_tokens,
        compact_tokens=args.compact_tokens,
        synthesis_tokens=args.synth_tokens,
        temperature_branch=args.temp_branch,
        temperature_synthesis=args.temp_synth,
    )
    trainer_cfg = TrainerConfig(model_name=args.model, prompt=prompt_cfg)
    reward = MathReward(RewardConfig())
    trainer = PaCoReTrainer(trainer_cfg, reward)
    trainer.train_epoch(dataset)
    trainer.save(Path(args.output))
