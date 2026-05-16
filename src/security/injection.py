"""Baseline prompt-injection detector.

This is not a bullet-proof solution — it is a first-pass filter for obvious
cases. Production systems typically add sanitization via a separate, dedicated
LLM classifier.
"""

import re


# Patterns commonly seen in injection attempts
INJECTION_PATTERNS = [
    r"\bignore\s+(previous|prior|above|all)\s+(instructions?|prompts?)\b",
    r"\bforget\s+(everything|previous|all\s+(instructions?|prompts?))\b",
    r"\bdisregard\s+(previous|prior|all|the)\s+(instructions?|prompts?)\b",
    r"\bnew\s+instructions?:\s*",
    r"\bsystem\s*:\s*",
    r"\b(you\s+are\s+now|act\s+as)\s+",
    # Russian-language variants
    r"\b(забудь|игнорируй|отмени)\s+(всё|все|предыдущие|инструкции|команды)\b",
    r"\bты\s+теперь\s+",
    r"\bновые\s+инструкции:\s*",
]


def is_likely_injection(text: str) -> bool:
    """Return True if the text contains classic prompt-injection markers."""
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True
    return False


def detection_reasons(text: str) -> list[str]:
    """Return the list of matched patterns (for logging on rejection)."""
    lower = text.lower()
    matches = []
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            matches.append(pattern)
    return matches
