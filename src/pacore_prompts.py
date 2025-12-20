"""
Prompt helpers and text utilities for the PaCoRe parallel reasoning pipeline.
The same model is used in three roles: branch generation, compaction, synthesis.
"""
from dataclasses import dataclass
from typing import List, Optional
import re


FINAL_TAG = "FINAL_ANSWER:"
INTERMEDIATE_TAG = "FINAL_INTERMEDIATE_ANSWER:"


@dataclass
class PaCoRePromptConfig:
    branches: int = 4
    branch_tokens: int = 256
    compact_tokens: int = 96
    synthesis_tokens: int = 256
    temperature_branch: float = 0.7
    temperature_synthesis: float = 0.4
    top_p: float = 0.9
    rounds: int = 1


def build_branch_prompt(problem: str) -> str:
    return (
        "You are a careful reasoning assistant.\n"
        "Solve the problem step by step. Keep derivations explicit.\n"
        f"Problem: {problem}\n\n"
        "End with a single line exactly like: FINAL_INTERMEDIATE_ANSWER: <best guess>"
    )


def build_compaction_prompt(trace: str, token_limit: int) -> str:
    return (
        "You are a concise log summarizer.\n"
        "Compress the reasoning trace into a short note with only key facts,"
        " derived results, and hypotheses. Remove filler and hesitation.\n"
        f"Target length: <= {token_limit} tokens.\n\n"
        "Reasoning trace:\n"
        f"{trace}\n\n"
        "Now produce the compact note."
    )


def build_synthesis_prompt(problem: str, notes: List[str]) -> str:
    formatted = "\n".join([f"[Note {i+1}] {note}" for i, note in enumerate(notes)])
    return (
        "You are an expert synthesizer.\n"
        "Given partial solution notes (some may be wrong), do:\n"
        "1) Cross-check and spot contradictions or mistakes.\n"
        "2) Combine correct ideas into one coherent plan.\n"
        "3) Produce a final, justified solution.\n"
        "If uncertain, choose the most plausible answer.\n\n"
        f"Problem: {problem}\n\n"
        f"Notes:\n{formatted}\n\n"
        "Write a single integrated solution.\n"
        "End with a single line exactly like: FINAL_ANSWER: <answer>"
    )


def _last_match_group(text: str, pattern: re.Pattern) -> Optional[str]:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def parse_tagged_answer(text: str, tag: str) -> Optional[str]:
    """Parse answers robustly.

    Models sometimes emit minor variations like `FINAL ANSWER:` vs `FINAL_ANSWER:`.
    This parser accepts spaces/underscores and returns the last occurrence.
    """
    tag_norm = tag.strip().rstrip(":")
    tag_norm = re.escape(tag_norm)
    # Allow spaces/underscores between words in the tag.
    tag_pattern = tag_norm.replace("_", r"[\s_]*")
    pattern = re.compile(tag_pattern + r"\s*:\s*(.+)", re.IGNORECASE)
    return _last_match_group(text, pattern)


def parse_final_answer(text: str) -> Optional[str]:
    return parse_tagged_answer(text, FINAL_TAG)


def parse_intermediate_answer(text: str) -> Optional[str]:
    return parse_tagged_answer(text, INTERMEDIATE_TAG)
