"""Post-hoc verification of LLM-produced citations against the source context."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    doc_id: int
    quote: str


def _normalize(s: str, normalize_ws: bool = True, case_sensitive: bool = False) -> str:
    if normalize_ws:
        s = re.sub(r"\s+", " ", s).strip()
    if not case_sensitive:
        s = s.lower()
    return s


def verify_citations(
    citations: list[Citation],
    context: dict[int, str],
) -> tuple[bool, list[Citation]]:
    invalid: list[Citation] = []
    for c in citations:
        if c.doc_id not in context:
            invalid.append(c)
            continue
        quote_norm = _normalize(c.quote)
        if not quote_norm:
            invalid.append(c)
            continue
        ctx_norm = _normalize(context[c.doc_id])
        if quote_norm not in ctx_norm:
            invalid.append(c)
    return (len(invalid) == 0, invalid)
