from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
URL_TOKEN = "urltoken"
USER_TOKEN = "usertoken"
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@[\w_]+", re.UNICODE)
_TOKEN_RE = re.compile(r"#[\w]+|[\w]+(?:['’][\w]+)?|[^\w\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    """Normalize markup, URLs, mentions and whitespace without removing local words."""
    text = html.unescape(str(text)).casefold()
    text = _URL_RE.sub(f" {URL_TOKEN} ", text)
    text = _MENTION_RE.sub(f" {USER_TOKEN} ", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def normalized_key(text: str) -> str:
    """Stable exact-text key for duplicate handling and leakage-safe grouping."""
    return " ".join(tokenize(text))


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        max_size: int = 30000,
        min_frequency: int = 1,
    ) -> "Vocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        # Deterministic ordering: frequency descending, then lexical tie-break.
        ordered = sorted(
            (item for item in counts.items() if item[1] >= min_frequency),
            key=lambda item: (-item[1], item[0]),
        )[: max(0, max_size - 2)]
        mapping = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        mapping.update({token: i + 2 for i, (token, _) in enumerate(ordered)})
        return cls(mapping)

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = [self.token_to_id.get(token, 1) for token in tokenize(text)]
        ids = ids[:max_length]
        return ids or [1]

    def to_json(self) -> dict[str, int]:
        return self.token_to_id

    @classmethod
    def from_json(cls, value: dict[str, int]) -> "Vocabulary":
        if value.get(PAD_TOKEN) != 0 or value.get(UNK_TOKEN) != 1:
            raise ValueError("Vocabulary must reserve <pad>=0 and <unk>=1")
        return cls({str(token): int(index) for token, index in value.items()})
