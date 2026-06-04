"""Baseline prompt-injection detector.

This is not a bullet-proof solution — it is a first-pass filter for obvious
cases. Production systems typically add sanitization via a separate, dedicated
LLM classifier.
"""

import re


# Building blocks. The override verbs ("ignore"/"disregard"/"forget") can be followed
# by several stacked qualifiers before the noun — the textbook string is "ignore ALL
# PREVIOUS instructions", with both a quantifier and a position word — so allow 1..3 of
# them rather than exactly one.
_SCOPE = r"(?:all|the|any|every|previous|prior|above|preceding|earlier|initial|original)"
_TARGET = r"(?:instructions?|prompts?|messages?|directions?|rules?|guidelines?|context|commands?)"

# Patterns commonly seen in injection attempts
INJECTION_PATTERNS = [
    rf"\bignore\s+(?:{_SCOPE}\s+){{1,3}}{_TARGET}\b",
    rf"\bdisregard\s+(?:{_SCOPE}\s+){{1,3}}{_TARGET}\b",
    rf"\bforget\s+(?:everything\b|(?:{_SCOPE}\s+){{1,3}}{_TARGET}\b)",
    r"\bnew\s+instructions?\s*:",
    r"\bsystem\s*:\s*",
    r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\b",
    # Russian-language variants
    r"\b(?:забудь|игнорируй|проигнорируй|отмени)\s+(?:всё|все|предыдущие|прежние|инструкции|команды|указания)\b",
    r"\bты\s+теперь\b",
    r"\bновые\s+инструкции\s*:",
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
