"""Fictional organization name generation and validation helpers."""

from __future__ import annotations

import random
import re


ORG_REQUIRED_TOKENS: dict[str, list[str]] = {
    "organization": ["Group", "Network", "Council", "Institute", "Association", "Center"],
    "military_org": ["Division", "Brigade", "Regiment", "Command", "Task Force", "Garrison"],
    "entreprise_org": [
        "Group",
        "Holdings",
        "Corporation",
        "Industries",
        "Systems",
        "Technologies",
        "Bank",
        "Logistics",
    ],
    "ngo": ["Foundation", "Alliance", "Association", "Initiative", "Council", "Network", "Relief"],
    "government_org": ["Ministry", "Department", "Agency", "Commission", "Bureau", "Office"],
    "educational_org": ["University", "College", "Institute", "Academy", "School"],
    "media_org": ["Times", "Post", "Gazette", "Herald", "News", "Radio", "Press", "Chronicle"],
}

_ORG_SYLLABLES = [
    "zor",
    "quar",
    "vex",
    "marn",
    "lith",
    "gry",
    "siv",
    "tarn",
    "wex",
    "brim",
    "krel",
    "nox",
    "ziv",
    "thur",
    "glan",
    "pry",
    "sorn",
    "vek",
    "drel",
    "yorn",
    "xal",
    "kiv",
    "rath",
    "vorn",
    "grel",
    "myr",
    "tov",
]


def _fictional_stem(rng: random.Random) -> str:
    for _ in range(200):
        parts = [rng.choice(_ORG_SYLLABLES) for _ in range(rng.choice([2, 3]))]
        stem = "".join(parts).capitalize()
        if len(stem) >= 4:
            return stem
    return "Zorvex"


def org_name_has_required_token(kind: str, name: str) -> bool:
    tokens = ORG_REQUIRED_TOKENS.get(kind)
    if not tokens:
        return True
    if not name:
        return False
    lower = name.lower()
    for token in tokens:
        pattern = r"\b" + re.escape(token.lower()) + r"\b"
        if re.search(pattern, lower):
            return True
    return False


def generate_org_name(kind: str, rng: random.Random, used: set[str] | None = None) -> str:
    tokens = ORG_REQUIRED_TOKENS.get(kind, ["Group"])
    patterns = ["{stem} {token}"]
    if kind in {"educational_org", "ngo"}:
        patterns.append("{token} of {stem}")
    if kind in {"government_org", "military_org"}:
        patterns.append("{stem} {token}")
    if kind in {"media_org"}:
        patterns.append("The {stem} {token}")

    used = used or set()
    for _ in range(400):
        stem = _fictional_stem(rng)
        token = rng.choice(tokens)
        pattern = rng.choice(patterns)
        name = pattern.format(stem=stem, token=token).strip()
        if name in used:
            continue
        if org_name_has_required_token(kind, name):
            return name
    return f"{_fictional_stem(rng)} {tokens[0]}"
