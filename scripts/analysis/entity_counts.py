"""Count replaced factual names in the paper's Infini-gram corpus index."""

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import requests
import yaml

INDEX = "v4_olmo-3-7b-pretrain-shared_olmo2"
API = "https://api.infini-gram.io/"
FIELDS = {
    "persons": {"full_name", "first_name", "last_name", "nationality", "ethnicity"},
    "places": {
        "name",
        "city",
        "state",
        "region",
        "country",
        "street",
        "natural_site",
        "continent",
        "demonym",
        "nationality",
    },
    "events": {"name"},
    "organizations": {"name"},
    "awards": {"name"},
    "legals": {"name", "reference_code"},
    "products": {"name"},
}


def named_entities(dataset):
    """Same field selection/deduplication as the original frequency experiment."""
    documents = defaultdict(dict)
    variants = defaultdict(set)
    for path in sorted((Path(dataset) / "FICTIONAL_DOCUMENTS/fictional").rglob("*.yaml")):
        payload = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        key = payload["document_theme"], payload["document_id"]
        variants[key].add(payload["document_variant_id"])
        for group, entities in (payload.get("replaced_factual_entities") or {}).items():
            for entity, fields in entities.items():
                for field, value in fields.items():
                    if field not in FIELDS.get(group, set()) or not isinstance(value, str):
                        continue
                    surface = " ".join(value.split())
                    if surface:
                        documents[key][(group, entity, field, surface)] = surface
    if len(documents) != 100 or any(v != {f"v{i:02d}" for i in range(1, 11)} for v in variants.values()):
        raise ValueError("Frequency counting requires all 100 fully replaced documents and ten variants")
    return documents


def query_count(surface, session):
    response = session.post(API, json={"index": INDEX, "query_type": "count", "query": surface}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    count = payload.get("count")
    if payload.get("error") or type(count) is not int or count < 0 or payload.get("approx"):
        raise ValueError(f"No exact count returned for {surface!r}; do not substitute missing counts with zero")
    return count


def collect_counts(dataset, output, *, execute=False):
    from .paper_reports import write_csv

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    documents = named_entities(dataset)
    surfaces = sorted({surface for entities in documents.values() for surface in entities.values()})
    if not execute:
        print(f"Would query {len(surfaces)} unique names in {INDEX}. Add --execute to contact Infini-gram.")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = output.with_suffix(".cache.jsonl")
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            row = json.loads(line)
            if row["index"] != INDEX or type(row["count"]) is not int or row["count"] < 0:
                raise ValueError("Invalid or mismatched frequency cache")
            if row["surface"] in cache and cache[row["surface"]] != row["count"]:
                raise ValueError("Conflicting counts in frequency cache")
            cache[row["surface"]] = row["count"]
    with requests.Session() as session, cache_path.open("a") as stream:
        for surface in surfaces:
            if surface not in cache:
                cache[surface] = query_count(surface, session)
                stream.write(json.dumps(dict(index=INDEX, surface=surface, count=cache[surface])) + "\n")
                stream.flush()
    rows = []
    for (theme, document), entities in sorted(documents.items()):
        counts = [cache[surface] for surface in entities.values()]
        rows.append(
            dict(
                index=INDEX,
                document_theme=theme,
                document_id=document,
                num_named_entities=len(counts),
                mean_named_entity_count=statistics.mean(counts),
                median_named_entity_count=statistics.median(counts),
                mean_log1p_named_entity_count=statistics.mean(math.log1p(c) for c in counts),
                zero_count_named_entities=sum(c == 0 for c in counts),
            )
        )
    write_csv(output, rows)
