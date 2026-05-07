
# Data Model for MEQA Annotation

This guide describes how to annotate the MEQA dataset with the purpose of replacing known entities with fictional ones while making sure to stay linguistically correct.

# Entity Taxonomy

Please use the following templatization format: `[text sequence; entity_id.attribute]`.

## Entity Types

The data model uses **14 entity types**:

1. **Person** (`person_X`)
2. **Place** (`place_X`)
3. **Event** (`event_X`)
4. **Military Org** (`military_org_X`)
5. **Entreprise Org** (`entreprise_org_X`)
6. **Ngo** (`ngo_X`)
7. **Government Org** (`government_org_X`)
8. **Educational Org** (`educational_org_X`)
9. **Media Org** (`media_org_X`)
10. **Temporal** (`temporal_X`)
11. **Number** (`number_X`)
12. **Award** (`award_X`)
13. **Legal** (`legal_X`)
14. **Product** (`product_X`)

Each entity has a unique identifier (e.g., `person_1`, `place_2`, `event_1`) that must remain consistent across the document, questions, and answers.

## Entity Attributes:

### 1. Person (`person_X`)

Individual human being mentioned in the document

| Attribute | Description | Example(s) |
|-----------|-------------|------------|
| `full_name` | Complete name of the person (first, middle, and/or last) | `John Smith`, `Marie Curie`, `Barack Hussein Obama` |
| `first_name` | Person's given name | `John`, `Marie`, `Sarah` |
| `last_name` | Person's family name | `Smith`, `Curie`, `Johnson` |
| `age` | Person's age in years | `25`, `45`, `67` |
| `gender` | Person's gender | `male`, `female` |
| `nationality` | Person's nationality or citizenship | `American`, `French`, `Syrian` |
| `ethnicity` | Person's ethnic or cultural background | `Hispanic`, `Asian`, `African American` |
| `subj_pronoun` | Subject pronoun (he/she/they) | `he`, `she`, `they` |
| `obj_pronoun` | Object pronoun (him/her/them) | `him`, `her`, `them` |
| `poss_det_pronoun` | Possessive determiner (his/her/their) | `his`, `her`, `their` |
| `poss_pro_pronoun` | Possessive pronoun (his/hers/theirs) | `his`, `hers`, `theirs` |
| `refl_pronoun` | Reflexive pronoun (himself/herself/themselves) | `himself`, `herself`, `themselves` |
| `honorific` | Title or honorific | `Mr.`, `Ms.` |
| `relationship` | Special attribute for relationships between people (use relationship.person_Y) | `mother`, `brother`, `son` |
| `middle_name` | Person's middle name | `Marie`, `Fitzgerald` |

Note: use relationship references as `person_X.relationship.person_Y` in annotations.

---

### 2. Place (`place_X`)

Geographic location or landmark

| Attribute | Description | Example(s) |
|-----------|-------------|------------|
| `city` | Name of a city or town | `New York`, `Paris`, `Tokyo` |
| `region` | Geographic region or area | `New England`, `Midwest`, `Provence` |
| `state` | State or province | `California`, `Ontario`, `Bavaria` |
| `country` | Nation or country | `United States`, `France`, `Japan` |
| `street` | Street name or address | `Main Street`, `5th Avenue`, `Baker Street` |
| `natural_site` | Natural landmark or feature | `Mount Everest`, `Amazon River`, `Grand Canyon` |
| `continent` | An entity referring to a continent | `Europe`, `Africa`, `Asia` |
| `demonym` | The demonym of a place entity | `Syrian`, `African`, `French` |

---

### 3. Event (`event_X`)

Named event or occurrence

| Attribute | Description | Example(s) |
|-----------|-------------|------------|
| `name` | Name of the event | `World War II`, `Olympic Games`, `Renaissance` |
| `type` | Type or category of event | `war`, `conference`, `festival` |

---

### 4. Military Org (`military_org_X`)

Military organization

| Attribute | Description | Example(s) |
|-----------|-------------|------------|
| `name` | Name of the military organization | `IRA`, `Royal Guard Command` |

---

### 5. Entreprise Org (`entreprise_org_X`)

Private/company organization (enterprise)

| Attribute | Description | Example |
|-----------|-------------|------------|
| `name` | Name of the enterprise organization | `Google` |

---

### 6. Ngo (`ngo_X`)

NGO / non-governmental organization

| Attribute | Description | Example |
|-----------|-------------|------------|
| `name` | Name of the NGO | `Amnesty International` |

---

### 7. Government Org (`government_org_X`)

Government institution or agency

| Attribute | Description | Example(s) |
|-----------|-------------|------------|
| `name` | Name of the government organization | `U.S. State Department`, `European Commission` |

---

### 8. Educational Org (`educational_org_X`)

Educational institution

| Attribute | Description | Example |
|-----------|-------------|------------|
| `name` | Name of the educational organization | `Harvard University` |

---

### 9. Media Org (`media_org_X`)

Media/journalism organization

| Attribute | Description | Example(s) |
|-----------|-------------|------------|
| `name` | Name of the media organization | `BBC`, `Reuters` |

---

### 10. Temporal (`temporal_X`)

Time-related information (dates, times, etc.)

| Attribute | Description | Example(s) |
|-----------|-------------|------------|
| `day` | Day of the week | `Monday`, `Friday`, `Sunday` |
| `date` | Full date | `January 1, 2024`, `2024-01-01` |
| `year` | Year | `2024`, `1999`, `1776` |
| `month` | Month name or number | `January`, `12`, `March` |
| `timestamp` | Specific time or timestamp | `3:30 PM`, `15:30`, `noon` |
| `day_of_month` | Day number within a month | `15`, `1st`, `31` |

---

### 11. Number (`number_X`)

Numerical values and quantities

| Attribute | Description | Example(s) |
|-----------|-------------|------------|
| `int` | Integer number | `42`, `100`, `1000` |
| `str` | Number expressed as text | `forty-two`, `one hundred`, `a thousand` |
| `float` | Decimal number | `3.14`, `99.9`, `0.5` |
| `fraction` | Fractional value | `1/2`, `three quarters`, `2/3` |
| `percent` | A number representing a percentage (must be between 0 and 100) | `20`, `10`, `100` |
| `proportion` | A number that represents a proportion (must be between 0.0 and 1.0) | `0.1`, `0.00012`, `1.0` |

---

### 12. Award (`award_X`)

An entity that represents any kind of awards

| Attribute | Description | Example(s) |
|-----------|-------------|------------|
| `name` | - | - |

---

### 13. Legal (`legal_X`)

Legal or regulatory instrument (law, directive, regulation, treaty, policy framework)

| Attribute | Description | Example(s) |
|-----------|-------------|------------|
| `name` | Legal or regulatory instrument (law, directive, regulation, treaty, policy framework) | `Markets in Financial Instruments Directive 2014`, `Capital Requirements Regulation`, `Payment Services Directive` |
| `reference_code` | Official legal citation or identifier | `2014/65/EU`, `(EU) No 575/2013`, `2015/2366` |

---

### 14. Product (`product_X`)

An entity describing a product

| Attribute | Description | Example(s) |
|-----------|-------------|------------|
| `name` | The name of the product as mentioned in the text | `iPhone`, `Eliquis (apixaban)` |

---

## [Rules](#rules)

Rules capture constraints between entities that must be preserved when generating documents with fictional entities. Rules are specified separately from entity annotations.

### Person Age Comparisons

| Rule Format | Example |
|-------------|---------|
| `person_X.age < person_Y.age` | `person_1.age < person_6.age` |
| `person_X.age > person_Y.age` | `person_6.age > person_1.age` |
| `person_X.age = person_Y.age` | `person_1.age = person_2.age` |

---

## Key points to keep in mind

1. **Variable Uniqueness**: Each unique entity must have a unique identifier. If "Dzhokhar Tsarnaev" is `person_1`, all mentions must use `person_1`.

2. **Attribute Consistency**: Attributes of the same entity must be consistent. If `person_1.full_name = "Dzhokhar Tsarnaev"`, then `person_1.first_name = "Dzhokhar"` and `person_1.last_name = "Tsarnaev"`.

3. **Pronoun Consistency**: Pronouns must match the gender and number of the person. For a masculine singular person: `subj_pronoun = "he"`, `obj_pronoun = "him"`, `poss_det_pronoun = "his"`.

4. **Document-Questions/Answers Consistency**: Entities in questions and answers must be consistent with those in the document.
