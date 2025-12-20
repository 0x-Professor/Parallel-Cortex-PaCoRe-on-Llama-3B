"""
PaCoRe inference pipeline: parallel branch generation -> compaction -> synthesis.
This uses a single HF causal LM (e.g., Llama-3.2-3B) with role prompts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from .pacore_prompts import (
    PaCoRePromptConfig,
    build_branch_prompt,
    build_compaction_prompt,
    build_synthesis_prompt,
    parse_final_answer,
    parse_intermediate_answer,
)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PaCoRePipelineConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    device: Optional[str] = None
    prompt: PaCoRePromptConfig = PaCoRePromptConfig()
    max_batch: int = 8  # how many prompts to batch at once


class PaCoRePipeline:
    """Runs the PaCoRe parallel reasoning protocol for inference."""

    def __init__(self, config: PaCoRePipelineConfig):
        self.config = config
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = config.device or _default_device()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=model_dtype,
            device_map="auto" if device != "cpu" else None,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            "Loaded model {} on {} (pad_token_id={})",
            config.model_name,
            device,
            self.tokenizer.pad_token_id,
        )

    @torch.no_grad()
    def _generate(self, prompts: List[str], max_new_tokens: int, temperature: float) -> List[str]:
        """Batch generate responses for a list of prompts."""
        outputs: List[str] = []
        for i in range(0, len(prompts), self.config.max_batch):
            chunk = prompts[i : i + self.config.max_batch]
            tokenized = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(
                self.model.device
            )
            generated = self.model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=self.config.prompt.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(decoded)
        return outputs

    @torch.no_grad()
    def run(self, problem: str) -> Dict[str, Any]:
        cfg = self.config.prompt

        branch_prompts = [build_branch_prompt(problem) for _ in range(cfg.branches)]
        branch_outputs = self._generate(branch_prompts, cfg.branch_tokens, cfg.temperature_branch)
        branch_answers = [parse_intermediate_answer(txt) for txt in branch_outputs]

        compact_prompts = [build_compaction_prompt(txt, cfg.compact_tokens) for txt in branch_outputs]
        compact_notes = self._generate(compact_prompts, cfg.compact_tokens, temperature=0.3)

        synth_prompt = build_synthesis_prompt(problem, compact_notes)
        synth_output = self._generate([synth_prompt], cfg.synthesis_tokens, cfg.temperature_synthesis)[0]
        final_answer = parse_final_answer(synth_output)

        return {
            "problem": problem,
            "branches": branch_outputs,
            "branch_answers": branch_answers,
            "compact_notes": compact_notes,
            "synthesis": synth_output,
            "final_answer": final_answer,
        }
