"""Fictional entity generation helpers for manual entity pools."""

from __future__ import annotations

from dataclasses import dataclass
import random


_NAME_SYLLABLES = [
    "jor",
    "lor",
    "vex",
    "tal",
    "mir",
    "zan",
    "kor",
    "vel",
    "rim",
    "sar",
    "quin",
    "dor",
    "fel",
    "gar",
    "lin",
    "mor",
    "nor",
    "pol",
    "rin",
    "tir",
    "vak",
    "zen",
    "bar",
    "del",
    "gor",
    "hal",
    "kin",
    "lor",
    "mur",
    "nel",
]

_DEMONYM_SYLLABLES = [
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

_DEMONYM_SUFFIXES = ["ian", "ese", "ish", "ite", "an", "ic", "oid", "ean"]

_REAL_DEMONYM_BLOCKLIST = {
    "american",
    "british",
    "chinese",
    "french",
    "german",
    "italian",
    "spanish",
    "russian",
    "indian",
    "japanese",
    "korean",
    "mexican",
    "canadian",
    "australian",
    "brazilian",
    "argentine",
    "greek",
    "turkish",
    "iranian",
    "iraqi",
    "israeli",
    "egyptian",
    "nigerian",
    "kenyan",
    "ethiopian",
    "swedish",
    "norwegian",
    "danish",
    "finnish",
    "polish",
    "ukrainian",
    "dutch",
    "belgian",
    "austrian",
    "swiss",
}


@dataclass(frozen=True)
class FictionalPlaceSpec:
    suffixes: list[str]
    patterns: list[str]


_PLACE_SPECS: dict[str, FictionalPlaceSpec] = {
    "city": FictionalPlaceSpec(
        suffixes=["ville", "ton", "burg", "polis", "stead", "ford", "mouth", "port"],
        patterns=["{stem}{suffix}", "{stem} City", "{stem} Town"],
    ),
    "town": FictionalPlaceSpec(
        suffixes=["ton", "burg", "stead", "ford"],
        patterns=["{stem}{suffix}", "{stem} Town"],
    ),
    "state": FictionalPlaceSpec(
        suffixes=["ia", "land", "mark", "shire"],
        patterns=["{stem}{suffix}", "State of {stem}", "{stem} State"],
    ),
    "country": FictionalPlaceSpec(
        suffixes=["ia", "land", "stan", "mark"],
        patterns=["{stem}{suffix}", "Republic of {stem}", "Kingdom of {stem}"],
    ),
    "region": FictionalPlaceSpec(
        suffixes=["lands", "ridge", "vale", "delta"],
        patterns=["{stem}{suffix}", "{stem} Region"],
    ),
    "province": FictionalPlaceSpec(
        suffixes=["shire", "vale", "march"],
        patterns=["{stem}{suffix}", "{stem} Province"],
    ),
    "island": FictionalPlaceSpec(
        suffixes=["isle", "cay", "island"],
        patterns=["{stem} {suffix}", "{stem}{suffix}"],
    ),
    "mountain": FictionalPlaceSpec(
        suffixes=["Peak", "Mount", "Ridge"],
        patterns=["{suffix} {stem}", "{stem} {suffix}"],
    ),
    "lake": FictionalPlaceSpec(
        suffixes=["Lake", "Lagoon"],
        patterns=["{suffix} {stem}", "{stem} {suffix}"],
    ),
    "street": FictionalPlaceSpec(
        suffixes=["Street", "Road", "Avenue", "Boulevard", "Lane", "Drive", "Way"],
        patterns=["{stem} {suffix}"],
    ),
}

_EVENT_TYPE_SUFFIXES = ["athon", "fest", "meet", "cup", "derby", "gala", "con"]
_EVENT_NAME_SUFFIXES = [
    "Jubilee",
    "Summit",
    "Incident",
    "Festival",
    "Expo",
    "Crisis",
    "Uprising",
    "Campaign",
    "Conference",
]
_AWARD_SUFFIXES = ["Prize", "Award", "Medal", "Cup", "Honor", "Distinction", "Trophy"]
_LEGAL_QUALIFIERS = [
    "Access",
    "Transit",
    "Research",
    "Security",
    "Health",
    "Trade",
    "Energy",
    "Civic",
    "Exchange",
    "Coordination",
]
_LEGAL_NOUNS = ["Act", "Charter", "Directive", "Framework", "Protocol", "Accord", "Code", "Statute", "Resolution"]
_PRODUCT_SUFFIXES = ["Core", "Suite", "Cloud", "Link", "One", "Flow", "Kit", "Works", "Drive", "Forge"]


def _fictional_stem(rng: random.Random, syllables: list[str]) -> str:
    for _ in range(200):
        parts = [rng.choice(syllables) for _ in range(rng.choice([2, 3]))]
        stem = "".join(parts).capitalize()
        if len(stem) >= 4:
            return stem
    return "Zorvex"


def fictional_name(rng: random.Random) -> str:
    return _fictional_stem(rng, _NAME_SYLLABLES)


def fictional_demonym(rng: random.Random) -> str:
    for _ in range(200):
        base = _fictional_stem(rng, _DEMONYM_SYLLABLES)
        suffix = rng.choice(_DEMONYM_SUFFIXES)
        value = f"{base}{suffix}"
        if value.lower() in _REAL_DEMONYM_BLOCKLIST:
            continue
        return value
    return "Zorvexian"


def name_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    a_l = a.lower()
    b_l = b.lower()
    if a_l == b_l:
        return True
    if a_l in b_l or b_l in a_l:
        return True
    if a_l[:3] == b_l[:3]:
        return True
    if a_l[-3:] == b_l[-3:]:
        return True
    if len(a_l) == len(b_l):
        diffs = sum(1 for x, y in zip(a_l, b_l) if x != y)
        if diffs <= 1:
            return True
    return False


def value_similar_to_names(value: str, person: dict) -> bool:
    if not value:
        return False
    val = str(value).lower()
    name_parts = []
    for attr in ("first_name", "last_name", "full_name"):
        v = person.get(attr)
        if v:
            name_parts.extend(str(v).lower().split())
    for name in name_parts:
        if not name:
            continue
        if val in name or name in val:
            return True
        if len(name) >= 3 and val.startswith(name[:3]):
            return True
    return False


def resample_demonym(person: dict, rng: random.Random) -> str:
    for _ in range(200):
        value = fictional_demonym(rng)
        if not value_similar_to_names(value, person):
            return value
    return fictional_demonym(rng)


def fictional_place(kind: str, rng: random.Random) -> str:
    spec = _PLACE_SPECS.get(kind)
    stem = _fictional_stem(rng, _DEMONYM_SYLLABLES)
    if not spec:
        return stem
    suffix = rng.choice(spec.suffixes)
    pattern = rng.choice(spec.patterns)
    return pattern.format(stem=stem, suffix=suffix).strip()


def fictional_event_type(rng: random.Random) -> str:
    stem = _fictional_stem(rng, _DEMONYM_SYLLABLES)
    suffix = rng.choice(_EVENT_TYPE_SUFFIXES)
    return f"{stem}{suffix}"


def fictional_event_name(rng: random.Random) -> str:
    stem = _fictional_stem(rng, _DEMONYM_SYLLABLES)
    suffix = rng.choice(_EVENT_NAME_SUFFIXES)
    if rng.random() < 0.35:
        return f"The {stem} {suffix}"
    return f"{stem} {suffix}"


def fictional_award_name(rng: random.Random) -> str:
    stem = _fictional_stem(rng, _DEMONYM_SYLLABLES)
    suffix = rng.choice(_AWARD_SUFFIXES)
    if rng.random() < 0.25:
        return f"The {stem} {suffix}"
    return f"{stem} {suffix}"


def fictional_legal_name(rng: random.Random) -> str:
    stem = _fictional_stem(rng, _DEMONYM_SYLLABLES)
    qualifier = rng.choice(_LEGAL_QUALIFIERS)
    noun = rng.choice(_LEGAL_NOUNS)
    if rng.random() < 0.35:
        return f"{qualifier} {noun} of {stem}"
    return f"{stem} {qualifier} {noun}"


def fictional_legal_reference_code(rng: random.Random) -> str:
    year = rng.randint(1995, 2049)
    major = rng.randint(1, 999)
    suffix = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
    return f"{year}/{major:03d}/{suffix}"


def fictional_product_name(rng: random.Random) -> str:
    stem = _fictional_stem(rng, _NAME_SYLLABLES)
    suffix = rng.choice(_PRODUCT_SUFFIXES)
    if rng.random() < 0.5:
        return f"{stem}{suffix}"
    return f"{stem} {suffix}"
