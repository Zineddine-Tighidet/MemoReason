You generate document-specific pools of fictional named-entity variants for benchmark construction.

Purpose:
- The benchmark replaces factual entities with fictional ones and then generates fictional document versions from those pools.
- Each required entity reference must receive its own list of fictional variants.
- The same document variant index is shared across all references generated in one response.
- If rules connect several references in this batch, variant `1` for all of those references must jointly satisfy the rules, variant `2` must jointly satisfy the rules, and so on up to variant `15`.
- Your output is not the final fictional document. It is the reference-specific candidate pool that later code will sample from.

What must stay outside the pool:
- Do NOT generate numbers.
- Do NOT generate dates, years, weekdays, months, timestamps, or any other temporals.
- Do NOT generate person gender attributes, pronouns, honorifics, or relationship fields.
- Those values are generated later by Python code with fixed seeds and rule checking.

Why the rules matter here:
- The rules describe constraints that the later document generator must preserve.
- Your pool should make those constraints easy to satisfy.
- Example: if a rule requires `person_1.nationality == place_2.demonym`, generate person nationalities and place demonyms that can be matched cleanly.
- Example: if a legal item needs a `name` and a `reference_code`, generate values that plausibly belong to the same fictional legal instrument.
- Ignore rules that only concern numbers, temporals, or other automatically generated fields.
- When you invent a place and a corresponding demonym or nationality adjective, they must belong to the same invented place.
- Do not pair a country, city, region, or state with a demonym that clearly belongs to some other invented place.
- If a place variant contains both `demonym` and `nationality`, keep them aligned unless the document explicitly requires different surface forms.

Entity taxonomy reference for this prompt:
- The list below is injected from the project taxonomy so the entity meanings and attribute meanings match the rest of the codebase exactly.
{{ENTITY_TAXONOMY_REFERENCE}}

Pool construction requirements:
- Every invented value must be fictional.
- In this benchmark, "fictional" means "non-existing": the value must not refer to a real entity that already exists.
- Treat any candidate that appears on Wikipedia or in a web search as invalid for this task.
- Before finalizing the pool, explicitly check whether the names you generated correspond to real entities. If your environment gives you web search or browsing tools, use them. If it does not, regenerate any candidate that seems plausibly real or widely used.
- Do NOT use ordinary attested first names, surnames, city names, country names, demonyms, award names, legal names, or organization names.
- Common human names are invalid even if you combine them with other fictional fields.
- If a candidate looks like a standard French, English, Spanish, Portuguese, German, Italian, Arabic, Slavic, or otherwise attested real-world name, reject it and invent a new one.
- Bad examples of invalid outputs: `Eliane`, `Lucien`, `Margaux`, `Renaud`, `Marcelo`, `Gaston`, `Adrienne`, `Henrik`, `Mallaby`, `Stembridge`, `Redwick`, `Marbleton`.
- Good outputs should feel pronounceable but unattested: they should read like plausible names while still looking clearly invented.
- Every invented name must be pronounceable for an English speaker.
- Follow ordinary English phonotactics. Avoid impossible consonant clusters, unreadable punctuation, excessive doubled letters, and fantasy-style spellings.
- Use ASCII only.
- Avoid accents, diacritics, emoji, and decorative punctuation.
- Avoid names that obviously match real well-known entities.
- Avoid names that are only tiny edits of famous real entities.
- Avoid slight edits of ordinary real first names or surnames.
- Avoid ordinary English surname or town-style endings such as `-ton`, `-bridge`, `-wick`, `-bury`, `-ford`, or `-by` when the result looks like an attested real place or family name.
- Avoid slight edits of real demonyms, historical labels, dynasties, eras, empires, or treaty names.
- Do not use standalone Roman numerals or ordinal dynastic labels as entity names.
- Do not invent names by taking a real root and adding a thin suffix like `-an`, `-ian`, `-ish`, `-ic`, or `-a`.
- Do not use generic suffix-marker tokens such as `Alt`, `Astra`, `Nova`, `Prime`, or `Sigma` anywhere in generated values.
- Avoid joke names, placeholders, and nonsense strings.
- Keep the pool diverse. Do not output many near-duplicates that only differ by one letter or one generic suffix.
- If a place entry contains several attributes, they must be internally coherent inside the same object.
- If a place entry contains both a place name and a demonym, the demonym must clearly match that exact fictional place.
- If an organization entry is tagged with a subtype, the name must sound plausible for that subtype.

Semantic consistency guardrails (critical):
- Do not generate names that semantically contradict explicit cues in the document text.
- Preserve ideological/polarity cues when those cues are explicitly stated.
- Avoid lexical markers that imply the opposite of the described role.
- If the text gives a clear stance, mission, alignment, or institutional function, generated names must remain compatible with that context.
- Use these examples as strict constraints:
  - If a political party is described as left-wing/progressive, do not generate a name that strongly signals right-wing/ultra-conservative/monarchist alignment.
  - If a party is described as conservative/right-wing, do not generate a name that strongly signals socialist/leftist alignment.
  - If an organization is described as humanitarian, relief-focused, or pacifist, do not generate a militaristic or combat-framed name.
  - If an organization is described as environmental/climate-focused, do not generate a name that suggests fossil-fuel expansion or anti-environment positioning.
  - If an entity is a court, ministry, or regulator, do not generate a name that sounds like a private company brand.
  - If an entity is a company/commercial operator, do not generate a name that sounds like a government ministry or tribunal.
  - If an entity is a university/school, do not generate a name that sounds like a bank, military unit, or political party.
  - If a media outlet is described as local/regional, do not generate a name implying global or official state-agency status unless the text supports it.
  - If the document is about countries, empires, wars, revolutions, or historical periods, do not output names that look like real-world historical labels or near-variants such as `Frankish`, `Gallican`, `Europan`, `Bourbon Restoration`, `Golden Era`, or bare numerals like `III`.

Reference-level requirements:
- allocate {{TARGET_CANDIDATES_PER_ENTITY}} fictional candidates for each unique entity reference requested below.
- Generate exactly {{TARGET_CANDIDATES_PER_ENTITY}} fictional variants for every required entity reference shown below.
- Treat the requested reference ids as opaque keys for this call. Do not rename them, renumber them, substitute different ids from the document, or output a nearby reference id.
- If this call requests `place_7` and `place_8`, then output exactly `place_7` and `place_8`. Outputting `place_2`, `place_4`, or any other reference id is invalid.
- Do not merge entity references together, even when they share the same type.
- Keep variants globally distinct across entity references of the same bucket. Do not reuse the same fictional person/place/event/etc. under two different reference ids.
- Only include the attributes required for that specific reference. Do not add unrelated optional attributes.
- For organization entities, write values into the taxonomy bucket that matches the annotation type.
- Within one response, list order matters: `variants[0]` across linked references describes one coherent fictional document version, `variants[1]` describes another, and so on.
- When linked references involve nationality or demonym fields, keep the same variant index aligned so the person/place pair still matches at that index.

Inputs:
- `document_id`: {{DOCUMENT_ID}}
- `document_theme`: {{DOCUMENT_THEME}}
- `annotated_document_excerpt_for_this_call`:
{{ANNOTATED_DOCUMENT}}

- `requested_reference_mentions`:
{{REQUESTED_REFERENCE_MENTIONS}}

- `rules_relevant_to_pool_generation`:
{{POOL_RELEVANT_RULES}}
  - Treat this as the actionable rule subset for pool construction.

- `required_entities_summary`:
{{REQUIRED_ENTITIES_SUMMARY}}

- `reference_ids_for_this_call`:
{{REFERENCE_IDS_FOR_THIS_CALL}}

- `reference_target_counts`:
{{REFERENCE_TARGET_COUNTS}}

Output format (STRICT YAML ONLY):
```yaml
persons:
  person_1:
    required_attributes:
      - full_name
      - first_name
    count: 15
    variants:
      - full_name: <fictional full name>
        first_name: <fictional first name>
  person_2:
    required_attributes:
      - full_name
    count: 15
    variants:
      - full_name: <fictional full name>
places:
  place_1:
    required_attributes:
      - country
      - demonym
    count: 15
    variants:
      - country: <fictional country>
        demonym: <fictional demonym>
events:
  event_1:
    required_attributes:
      - name
    count: 15
    variants:
      - name: <fictional event name>
        type: <event type if needed>
military_orgs:
  military_org_1:
    required_attributes:
      - name
    count: 15
    variants:
      - name: <fictional military organization name>
entreprise_orgs:
  entreprise_org_1:
    required_attributes:
      - name
    count: 15
    variants:
      - name: <fictional enterprise organization name>
ngos:
  ngo_1:
    required_attributes:
      - name
    count: 15
    variants:
      - name: <fictional NGO name>
government_orgs:
  government_org_1:
    required_attributes:
      - name
    count: 15
    variants:
      - name: <fictional government organization name>
educational_orgs:
  educational_org_1:
    required_attributes:
      - name
    count: 15
    variants:
      - name: <fictional educational organization name>
media_orgs:
  media_org_1:
    required_attributes:
      - name
    count: 15
    variants:
      - name: <fictional media organization name>
awards:
  award_1:
    required_attributes:
      - name
    count: 15
    variants:
      - name: <fictional award name>
legals:
  legal_1:
    required_attributes:
      - name
      - reference_code
    count: 15
    variants:
      - name: <fictional legal entity name>
        reference_code: <fictional legal code if needed>
products:
  product_1:
    required_attributes:
      - name
    count: 15
    variants:
      - name: <fictional product name>
```

Output rules:
1. Return YAML only.
2. Output only the buckets and entity references requested in `required_entities_summary`.
3. Output only the reference ids listed in `reference_ids_for_this_call`. Any other id is invalid.
4. Keep only relevant keys in each variant object. Do not write null values.
5. For organization buckets, write only `{name: ...}` entries inside each variant. The bucket name already carries the taxonomy type.
6. Set `count` to the number of distinct valid variants you actually provide for that reference.
7. Respect `reference_target_counts` and aim to make every requested `count` equal `15`.
8. Use the rules to make the pool compatible with the later replacement step.
9. Prefer varied, clean fictional values over tiny spelling variations of the same item.
