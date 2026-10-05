from __future__ import annotations

import re

_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_match_text(value: str) -> str:
    return " ".join(_NON_WORD.sub(" ", value.casefold()).split())


def contains_alias(text: str, alias: str) -> bool:
    normalized_text = f" {normalize_match_text(text)} "
    normalized_alias = normalize_match_text(alias)
    if len(normalized_alias) < 3:
        return False
    return f" {normalized_alias} " in normalized_text
