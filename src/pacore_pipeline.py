"""
PaCoRe inference pipeline: parallel branch generation -> compaction -> synthesis.
This uses a single HF causal LM (e.g., Llama-3.2-3B) with role prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
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


def _normalize_device(device: Optional[str]) -> str:
    if not device:
        return "auto"
    return device.strip().lower()


def _resolve_device(device: Optional[str]) -> str:
    dev = _normalize_device(device)
    if dev == "auto":
        return _default_device()
    if dev in {"cpu", "cuda", "mps"}:
        return dev
    raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cpu, cuda, mps")


@dataclass
class PaCoRePipelineConfig:
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    device: Optional[str] = None
    trust_remote_code: bool = False
    hf_token: Optional[str] = None
    prompt: PaCoRePromptConfig = field(default_factory=PaCoRePromptConfig)
    max_batch: int = 8  # how many prompts to batch at once
    # Regex-based stopping is expensive because it requires decoding every step.
    # Keep it off by default; rely on max_new_tokens + tagged parsing.
    regex_stopping: bool = False


class PaCoRePipeline:
    """Runs the PaCoRe parallel reasoning protocol for inference."""

    def __init__(self, config: PaCoRePipelineConfig):
        self.config = config
        device = _resolve_device(config.device)
        if device == "cuda":
            model_dtype = torch.bfloat16
        elif device == "mps":
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32

        # Prefer an explicit token (config) and then fall back to environment variables.
        # NOTE: `huggingface_hub.HfFolder.get_token()` only checks cached CLI logins; it
        # does not reflect tokens injected via environment variables.
        hf_token = (
            (config.hf_token or "").strip() or
            (os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip() or
            (os.environ.get("HF_TOKEN") or "").strip()
        )
        hf_token = hf_token or None

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
            token=hf_token,
        )
        # Decoder-only LMs should use left-padding for correct generation when batching.
        self.tokenizer.padding_side = "left"
        self._has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        # Newer Transformers prefers `dtype=`; older versions used `torch_dtype=`.
        # Use a small compatibility shim so users don't see deprecation warnings.
        def _from_pretrained_with_dtype(**kwargs):
            try:
                return AutoModelForCausalLM.from_pretrained(**kwargs)
            except TypeError:
                dtype = kwargs.pop("dtype", None)
                if dtype is not None:
                    kwargs["torch_dtype"] = dtype
                return AutoModelForCausalLM.from_pretrained(**kwargs)

        self.model = _from_pretrained_with_dtype(
            pretrained_model_name_or_path=config.model_name,
            dtype=model_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=config.trust_remote_code,
            token=hf_token,
        )
        if device in {"cpu", "mps"}:
            self.model.to(device)
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
            if stop_regex and bool(self.config.regex_stopping):
                stopping = StoppingCriteriaList(
                    [_StopOnCompletionRegex(self.tokenizer, input_lengths, stop_regex)]
                )

            do_sample = temperature is not None and float(temperature) > 0.0

            generate_kwargs = {
                **tokenized,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": (float(temperature) if do_sample else None),
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "stopping_criteria": stopping,
            }
            if do_sample:
                generate_kwargs["top_p"] = self.config.prompt.top_p

            generated = self.model.generate(
                **generate_kwargs,
            )

            # Decode only the generated completion (exclude the prompt), even with padding.
            for row, in_len in zip(generated, input_lengths.tolist()):
                completion_ids = row[int(in_len) :]
                outputs.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip())
        return outputs

    @torch.no_grad()
    def run(self, problem: str) -> Dict[str, Any]:
        cfg = self.config.prompt

        if cfg.branches < 1:
            raise ValueError("prompt.branches must be >= 1")

        branch_prompts = [build_branch_prompt(problem) for _ in range(cfg.branches)]
        branch_outputs = self._generate(
            branch_prompts,
            cfg.branch_tokens,
            cfg.temperature_branch,
            stop_regex=r"FINAL[\s_]*INTERMEDIATE[\s_]*ANSWER\s*:\s*\S",
        )
        branch_answers = [parse_intermediate_answer(txt) for txt in branch_outputs]

        # If we have tagged intermediate answers, compute a simple consensus.
        non_empty_branch_answers = [a for a in branch_answers if a is not None and str(a).strip()]
        consensus_answer: Optional[str] = None
        if non_empty_branch_answers:
            counts: Dict[str, int] = {}
            for a in non_empty_branch_answers:
                key = str(a).strip()
                counts[key] = counts.get(key, 0) + 1
            consensus_answer = max(counts.items(), key=lambda kv: kv[1])[0]

        # Build compact notes. When possible, do it deterministically using the tagged
        # intermediate answer so small token budgets don't drop the key result.
        compact_notes: List[str] = []
        missing_compaction_indices: List[int] = []
        for idx, (trace, ans) in enumerate(zip(branch_outputs, branch_answers)):
            if ans is not None and str(ans).strip():
                compact_notes.append(f"FINAL_INTERMEDIATE_ANSWER: {str(ans).strip()}")
            else:
                compact_notes.append("")
                missing_compaction_indices.append(idx)

        if missing_compaction_indices:
            compact_prompts = [
                build_compaction_prompt(branch_outputs[i], cfg.compact_tokens)
                for i in missing_compaction_indices
            ]
            generated_notes = self._generate(compact_prompts, cfg.compact_tokens, temperature=0.3)
            for i, note in zip(missing_compaction_indices, generated_notes):
                compact_notes[i] = note

        synth_prompt = build_synthesis_prompt(problem, compact_notes, candidate_answer=consensus_answer)
        synth_output = self._generate(
            [synth_prompt],
            cfg.synthesis_tokens,
            cfg.temperature_synthesis,
            stop_regex=r"FINAL[\s_]*ANSWER\s*:\s*\S",
        )[0]

        final_answer = parse_final_answer(synth_output)
        # Prefer the branch consensus if synthesis is missing/truncated or disagrees.
        if consensus_answer is not None:
            if final_answer is None or str(final_answer).strip() != str(consensus_answer).strip():
                final_answer = consensus_answer
        if final_answer is None:
            final_answer = synth_output.strip() or None

        return {
            "problem": problem,
            "branches": branch_outputs,
            "branch_answers": branch_answers,
            "compact_notes": compact_notes,
            "synthesis": synth_output,
            "final_answer": final_answer,
        }
