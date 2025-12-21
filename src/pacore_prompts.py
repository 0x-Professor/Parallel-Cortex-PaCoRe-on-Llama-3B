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
        "Answer concisely and follow the required tag format.\n"
        "Do NOT write code. Do NOT use fenced code blocks.\n\n"
        f"Problem: {problem}\n\n"
        "IMPORTANT: Put the answer tag FIRST so it is not lost if output is truncated.\n"
        "Line 1 must be exactly: FINAL_INTERMEDIATE_ANSWER: <answer>\n"
        "Then give 1-4 short lines of reasoning.\n"
        "Finally, repeat the same tag on the last line exactly: FINAL_INTERMEDIATE_ANSWER: <answer>"
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
        "You are an expert solver.\n"
        "Use the notes ONLY as internal guidance. Do NOT mention the notes or the protocol.\n"
        "Be direct and concise.\n\n"
        f"Problem: {problem}\n\n"
        f"Notes:\n{formatted}\n\n"
        "IMPORTANT: Put the answer tag FIRST so it is not lost if output is truncated.\n"
        "Line 1 must be exactly: FINAL_ANSWER: <answer>\n"
        "Then give a brief justification (1-5 short lines).\n"
        "Finally, repeat the same tag on the last line exactly: FINAL_ANSWER: <answer>"
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
