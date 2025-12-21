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


def build_branch_prompt(problem: str, round_num: int = 1, prior_answer: Optional[str] = None) -> str:
    """Build a branch prompt with optional multi-round context."""
    prior_block = ""
    if prior_answer is not None and round_num > 1:
        prior_block = (
            f"\nYour previous answer was: {prior_answer}\n"
            "Double-check it. If correct, confirm. If wrong, fix it.\n"
        )
    return (
        "You are an expert problem solver with exceptional reasoning skills.\n"
        "Think step by step. Break the problem into parts. Show your work.\n"
        "Do NOT write code. Do NOT use fenced code blocks.\n\n"
        f"Problem: {problem}\n"
        f"{prior_block}\n"
        "INSTRUCTIONS:\n"
        "1. First, understand what the problem is asking.\n"
        "2. Identify the key information and constraints.\n"
        "3. Choose an appropriate method/formula.\n"
        "4. Execute step by step, showing calculations.\n"
        "5. Verify your answer makes sense.\n\n"
        "FORMAT: Put the answer tag FIRST (in case output is truncated).\n"
        "Line 1: FINAL_INTERMEDIATE_ANSWER: <answer>\n"
        "Then 2-6 lines of step-by-step reasoning.\n"
        "Last line: FINAL_INTERMEDIATE_ANSWER: <answer>"
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


def build_synthesis_prompt(problem: str, notes: List[str], candidate_answer: Optional[str] = None) -> str:
    formatted = "\n".join([f"[Note {i+1}] {note}" for i, note in enumerate(notes)])
    candidate_block = ""
    if candidate_answer is not None and str(candidate_answer).strip():
        candidate_block = (
            f"\nCandidate answer (verify carefully): {str(candidate_answer).strip()}\n"
        )
    return (
        "You are an expert problem solver and verifier.\n"
        "Multiple reasoning attempts have been made. Your job is to:\n"
        "1. Cross-check the candidate answers for correctness.\n"
        "2. Identify any errors in reasoning.\n"
        "3. Produce the CORRECT final answer.\n\n"
        f"Problem: {problem}\n\n"
        f"Reasoning attempts:\n{formatted}\n"
        f"{candidate_block}\n"
        "VERIFICATION STEPS:\n"
        "- If answers agree, verify by re-solving quickly.\n"
        "- If answers disagree, identify which reasoning is correct.\n"
        "- Check for arithmetic errors, sign errors, or misunderstandings.\n\n"
        "FORMAT: Put the answer tag FIRST (in case output is truncated).\n"
        "Line 1: FINAL_ANSWER: <answer>\n"
        "Then 2-5 lines of verification/justification.\n"
        "Last line: FINAL_ANSWER: <answer>"
    )


def build_verification_prompt(problem: str, proposed_answer: str) -> str:
    """Build a prompt for verifying/double-checking an answer."""
    return (
        "You are a careful verifier. Check if the proposed answer is correct.\n\n"
        f"Problem: {problem}\n"
        f"Proposed answer: {proposed_answer}\n\n"
        "TASK:\n"
        "1. Re-solve the problem independently using a different method if possible.\n"
        "2. Compare your result with the proposed answer.\n"
        "3. If they match, confirm. If not, provide the correct answer.\n\n"
        "FORMAT:\n"
        "Line 1: FINAL_ANSWER: <answer>\n"
        "Then 2-4 lines explaining your verification.\n"
        "Last line: FINAL_ANSWER: <answer>"
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
