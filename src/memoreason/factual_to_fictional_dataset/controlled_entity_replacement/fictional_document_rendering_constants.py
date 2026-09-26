# ruff: noqa: RUF001
"""Constants used while rendering fictional document variants."""

import re


_GENDER_NOUN_CHILD_SINGULAR = frozenset({"boy", "girl"})
_GENDER_NOUN_CHILD_PLURAL = frozenset({"boys", "girls"})
_GENDER_NOUN_ADULT_SINGULAR = frozenset({"man", "woman", "guy", "lady"})
_GENDER_NOUN_ADULT_PLURAL = frozenset({"men", "women", "guys", "ladies"})
_GENDER_ADJECTIVES_SINGULAR = frozenset({"male", "female"})
_GENDER_ADJECTIVES_PLURAL = frozenset({"males", "females"})
_ARTICLELESS_NUMBER_WORDS = frozenset(
    {
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
    }
)
_GLOBAL_LITERAL_REWRITE_ATTRS = frozenset(
    {
        "first_name",
        "last_name",
        "full_name",
        "name",
        "nationality",
        "demonym",
        "country",
        "state",
        "region",
        "continent",
        "city",
        "natural_site",
    }
)
_DUPLICATE_PAREN_PATTERN = re.compile(r"\b(?P<name>[A-Z][A-Za-z0-9'’\" -]{2,}?)\s*\(\s*(?P=name)\s*\)")
_BIRTH_YEAR_PATTERN = re.compile(r"\bborn\b[^.]{0,80}?(\d{4})", re.IGNORECASE)
_AGE_IN_YEAR_PATTERN = re.compile(r"\bat age\s+(\d{1,3})\s+in\s+((?:[A-Za-z]+\s+)?)(\d{4})\b", re.IGNORECASE)
_DUPLICATE_SUFFIXES: tuple[str, ...] = (
    "region",
    "city",
    "state",
    "country",
    "province",
    "prefecture",
    "county",
    "ocean",
    "sea",
    "gulf",
    "bay",
    "river",
    "peninsula",
    "island",
    "islands",
    "mountain",
    "mountains",
    "lake",
)
_DEFINITE_ARTICLE_KEEP_TOKENS = frozenset(
    {
        "academy",
        "agency",
        "authority",
        "bank",
        "bay",
        "bureau",
        "channel",
        "city",
        "coast",
        "command",
        "commission",
        "commonwealth",
        "conference",
        "council",
        "county",
        "department",
        "desert",
        "dominions",
        "emirates",
        "federation",
        "framework",
        "gulf",
        "hospital",
        "institute",
        "island",
        "islands",
        "isle",
        "isles",
        "kingdom",
        "lake",
        "ministry",
        "mountain",
        "mountains",
        "ocean",
        "office",
        "peninsula",
        "plant",
        "plan",
        "power",
        "process",
        "program",
        "project",
        "protocol",
        "prefecture",
        "province",
        "region",
        "republic",
        "river",
        "sea",
        "state",
        "states",
        "station",
        "strait",
        "union",
        "university",
    }
)
_DEFINITE_ARTICLE_SINGLETONS = frozenset({"bahamas", "gambia", "netherlands", "philippines"})
