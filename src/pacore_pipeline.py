"""
PaCoRe inference pipeline: parallel branch generation -> compaction -> synthesis.
This uses a single HF causal LM (e.g., Llama-3.2-3B) with role prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
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
    trust_remote_code: bool = False
    prompt: PaCoRePromptConfig = field(default_factory=PaCoRePromptConfig)
    max_batch: int = 8  # how many prompts to batch at once


class PaCoRePipeline:
    """Runs the PaCoRe parallel reasoning protocol for inference."""

    def __init__(self, config: PaCoRePipelineConfig):
        self.config = config
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = config.device or _default_device()

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        # Decoder-only LMs should use left-padding for correct generation when batching.
        self.tokenizer.padding_side = "left"
        self._has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=model_dtype,
            device_map="auto" if device != "cpu" else None,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            "Loaded model {} on {} (pad_token_id={})",
            config.model_name,
            device,
            self.tokenizer.pad_token_id,
        )

    def _format_prompt(self, prompt: str) -> str:
        # Instruct/chat models (Llama Instruct, Qwen Instruct, etc.) work much better
        # when you use the model's chat template.
        if self._has_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    @torch.no_grad()
    def _generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        stop_regex: Optional[str] = None,
    ) -> List[str]:
        """Batch generate responses for a list of prompts."""

        class _StopOnCompletionRegex(StoppingCriteria):
            def __init__(self, tokenizer, input_lengths, pattern: str):
                self._tokenizer = tokenizer
                self._input_lengths = input_lengths
                self._pattern = pattern

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:  # type: ignore[override]
                # Stop when *any* sample matches; HF will stop the whole batch.
                # We keep batches small (max_batch) so this is OK for demos.
                for row, in_len in zip(input_ids, self._input_lengths.tolist()):
                    completion_ids = row[int(in_len) :]
                    text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
                    if re.search(self._pattern, text, flags=re.IGNORECASE | re.DOTALL):
                        return True
                return False

        outputs: List[str] = []
        for i in range(0, len(prompts), self.config.max_batch):
            chunk = [self._format_prompt(p) for p in prompts[i : i + self.config.max_batch]]
            tokenized = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).to(
                self.model.device
            )
            input_lengths = tokenized["attention_mask"].sum(dim=1)

            stopping: Optional[StoppingCriteriaList] = None
            if stop_regex:
                stopping = StoppingCriteriaList([
                    _StopOnCompletionRegex(self.tokenizer, input_lengths, stop_regex)
                ])

            generated = self.model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=self.config.prompt.top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=stopping,
            )

            # Decode only the generated completion (exclude the prompt), even with padding.
            for row, in_len in zip(generated, input_lengths.tolist()):
                completion_ids = row[int(in_len) :]
                outputs.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
        return outputs

    @torch.no_grad()
    def run(self, problem: str) -> Dict[str, Any]:
        cfg = self.config.prompt

        branch_prompts = [build_branch_prompt(problem) for _ in range(cfg.branches)]
        branch_outputs = self._generate(
            branch_prompts,
            cfg.branch_tokens,
            cfg.temperature_branch,
            stop_regex=r"FINAL[\s_]*INTERMEDIATE[\s_]*ANSWER\s*:\s*\S",
        )
        branch_answers = [parse_intermediate_answer(txt) for txt in branch_outputs]

        compact_prompts = [build_compaction_prompt(txt, cfg.compact_tokens) for txt in branch_outputs]
        compact_notes = self._generate(compact_prompts, cfg.compact_tokens, temperature=0.3)

        synth_prompt = build_synthesis_prompt(problem, compact_notes)
        synth_output = self._generate(
            [synth_prompt],
            cfg.synthesis_tokens,
            cfg.temperature_synthesis,
            stop_regex=r"FINAL[\s_]*ANSWER\s*:\s*\S",
        )[0]
        final_answer = parse_final_answer(synth_output)
        if final_answer is None:
            # Fallback: return the trimmed synthesis output so callers always get something usable.
            final_answer = synth_output.strip() or None

        return {
            "problem": problem,
            "branches": branch_outputs,
            "branch_answers": branch_answers,
            "compact_notes": compact_notes,
            "synthesis": synth_output,
            "final_answer": final_answer,
        }
