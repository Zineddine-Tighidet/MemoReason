"""
Data models: entities, documents, questions, rules.

  - Entity types: PersonEntity, PlaceEntity, EventEntity, OrganizationEntity,
    AwardEntity, LegalEntity, ProductEntity, TemporalEntity, NumberEntity
  - EntityCollection: container for all entities in a document
  - EntityPool: manual pools for sampling
  - Question, AnnotatedDocument, FictionalDocument: document/question structures
  - Rule, RuleType, AnnotationReference: rules and annotation refs
"""

from __future__ import annotations

import builtins
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .organization_types import ORG_ENTITY_TYPES


# Entity Types
class PersonEntity(BaseModel):
    """Person entity with all possible attributes"""
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    nationality: Optional[str] = None
    ethnicity: Optional[str] = None
    honorific: Optional[str] = None  # Mr, Ms, Mrs, etc.
    subj_pronoun: Optional[str] = None
    obj_pronoun: Optional[str] = None
    poss_det_pronoun: Optional[str] = None
    poss_pro_pronoun: Optional[str] = None
    refl_pronoun: Optional[str] = None
    relationship: Optional[str] = None  # Generic relationship for simple cases
    relationships: Optional[Dict[str, str]] = Field(default_factory=dict)  # person_id -> relationship
    middle_name: Optional[str] = None


class PlaceEntity(BaseModel):
    """Place entity with all possible attributes"""
    city: Optional[str] = None
    region: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    street: Optional[str] = None
    natural_site: Optional[str] = None
    continent: Optional[str] = None
    demonym: Optional[str] = None
    nationality: Optional[str] = None  # Adjective form (e.g. Syrian, French)


class EventEntity(BaseModel):
    """Event entity with all possible attributes"""
    name: Optional[str] = None
    type: Optional[str] = None


class OrganizationEntity(BaseModel):
    """Organization entity with an explicit subtype and surface name."""
    name: Optional[str] = None
    organization_kind: Optional[str] = None


class AwardEntity(BaseModel):
    """Award entity."""
    name: Optional[str] = None


class LegalEntity(BaseModel):
    """Legal or regulatory instrument."""
    name: Optional[str] = None
    reference_code: Optional[str] = None


class ProductEntity(BaseModel):
    """Product entity."""
    name: Optional[str] = None


class TemporalEntity(BaseModel):
    """Temporal entity with all possible attributes"""
    day: Optional[str] = None
    date: Optional[str] = None
    year: Optional[int] = None
    month: Optional[str] = None
    day_of_month: Optional[int] = None  # Numeric day (e.g. 21)
    timestamp: Optional[str] = None


class NumberEntity(BaseModel):
    """Number entity with all possible attributes"""
    int: Optional[builtins.int] = None
    str: Optional[builtins.str] = None
    float: Optional[builtins.float] = None
    fraction: Optional[builtins.str] = None
    percent: Optional[builtins.float] = None
    proportion: Optional[builtins.float] = None
    int_surface_format: Optional[builtins.str] = None
    str_surface_format: Optional[builtins.str] = None


# Mapping from canonical (singular) entity type keys to the attribute name on EntityCollection.
# Accepts both British and American spellings for organization/organisation.
ENTITY_TYPE_TO_ATTR: Dict[str, str] = {
    "person": "persons",
    "place": "places",
    "event": "events",
    "organization": "organizations",
    "award": "awards",
    "legal": "legals",
    "product": "products",
    "temporal": "temporals",
    "number": "numbers",
}
for _organization_type in sorted(ORG_ENTITY_TYPES):
    ENTITY_TYPE_TO_ATTR[_organization_type] = "organizations"

# Mapping from canonical (singular) entity type keys to the corresponding model class.
ENTITY_TYPE_TO_CLASS: Dict[str, type] = {
    "person": PersonEntity,
    "place": PlaceEntity,
    "event": EventEntity,
    "organization": OrganizationEntity,
    "award": AwardEntity,
    "legal": LegalEntity,
    "product": ProductEntity,
    "temporal": TemporalEntity,
    "number": NumberEntity,
}
for _organization_type in sorted(ORG_ENTITY_TYPES):
    ENTITY_TYPE_TO_CLASS[_organization_type] = OrganizationEntity


# Container for all entities
class EntityCollection(BaseModel):
    """Collection of all entities for a document."""
    persons: Dict[str, PersonEntity] = Field(default_factory=dict)  # person_1, person_2, ...
    places: Dict[str, PlaceEntity] = Field(default_factory=dict)
    events: Dict[str, EventEntity] = Field(default_factory=dict)
    organizations: Dict[str, OrganizationEntity] = Field(default_factory=dict)
    awards: Dict[str, AwardEntity] = Field(default_factory=dict)
    legals: Dict[str, LegalEntity] = Field(default_factory=dict)
    products: Dict[str, ProductEntity] = Field(default_factory=dict)
    temporals: Dict[str, TemporalEntity] = Field(default_factory=dict)
    numbers: Dict[str, NumberEntity] = Field(default_factory=dict)

    def get_collection(self, entity_type: str) -> Dict[str, Any]:
        """Return the entity dict for one canonical entity type key.

        Raises ValueError for unknown entity types so typos are caught immediately
        rather than silently producing wrong results.
        """
        attr = ENTITY_TYPE_TO_ATTR.get(entity_type)
        if attr is None:
            raise ValueError(f"Unknown entity type: {entity_type!r}")
        return getattr(self, attr)

    def add_entity(self, entity_type: str, entity_id: str, entity: Any) -> None:
        """Add an entity by canonical type key."""
        self.get_collection(entity_type)[entity_id] = entity

    def update_entity_attribute(self, entity_type: str, entity_id: str, attribute: str, value: Any) -> bool:
        """Update a single attribute on an existing entity, reconstructing it from its model class.

        Returns True if the entity was found and updated, False otherwise.
        """
        coll = self.get_collection(entity_type)
        if entity_id not in coll:
            return False
        entity_cls = ENTITY_TYPE_TO_CLASS.get(entity_type)
        if entity_cls is None:
            raise ValueError(f"Unknown entity type: {entity_type!r}")
        entity_dict = coll[entity_id].model_dump()
        entity_dict[attribute] = value
        coll[entity_id] = entity_cls(**entity_dict)
        return True

    def merge_from(self, other: "EntityCollection") -> None:
        """Merge all entities from *other* into this collection (other takes precedence on conflict)."""
        self.persons.update(other.persons)
        self.places.update(other.places)
        self.events.update(other.events)
        self.organizations.update(other.organizations)
        self.awards.update(other.awards)
        self.legals.update(other.legals)
        self.products.update(other.products)
        self.numbers.update(other.numbers)
        self.temporals.update(other.temporals)


# Entity Pool for manual entry (i.e. the sampling sets are manually written by human annotators)
# The rest of entities are auto-generated making sure that the constraints are satisfied.
class EntityPool(BaseModel):
    """Pool of entities to sample from (manual lists)"""
    persons: List[PersonEntity] = Field(default_factory=list)
    places: List[PlaceEntity] = Field(default_factory=list)
    events: List[EventEntity] = Field(default_factory=list)
    organizations: List[OrganizationEntity] = Field(default_factory=list)
    military_orgs: List[OrganizationEntity] = Field(default_factory=list)
    entreprise_orgs: List[OrganizationEntity] = Field(default_factory=list)
    ngos: List[OrganizationEntity] = Field(default_factory=list)
    government_orgs: List[OrganizationEntity] = Field(default_factory=list)
    educational_orgs: List[OrganizationEntity] = Field(default_factory=list)
    media_orgs: List[OrganizationEntity] = Field(default_factory=list)
    awards: List[AwardEntity] = Field(default_factory=list)
    legals: List[LegalEntity] = Field(default_factory=list)
    products: List[ProductEntity] = Field(default_factory=list)


# Annotation reference
class AnnotationReference(BaseModel):
    """Represents a single annotation like [text; entity_id.attribute]"""
    original_text: str
    entity_id: str
    attribute: Optional[str] = None
    start_pos: int # position of the opening bracket
    end_pos: int # position of the closing bracket


# Rule types
class RuleType(str, Enum):
    COMPARISON = "comparison"  # <, >, =, <=, >= (with or without arithmetic)


class Rule(BaseModel):
    """Represents a constraint rule"""
    raw_rule: str
    rule_type: RuleType
    left_operand: str
    operator: str
    right_operand: Union[str, int, float]


# Question and Answer
class Question(BaseModel):
    """Question with answer"""
    question_id: str
    question: str
    answer: str  # Can be expression or literal value
    question_type: Optional[str] = None  # e.g., "extractive", "temporal", "inference"
    answer_type: Optional[str] = None  # one of: variant, invariant, refusal
    reasoning_chain: List[str] = Field(default_factory=list)
    accepted_answer_overrides: List[str] = Field(default_factory=list)


class ImplicitRule(BaseModel):
    """Structured implicit range rule shown in the annotation UI."""

    entity_ref: str
    lower_bound: builtins.float
    upper_bound: builtins.float
    factual_value: builtins.float
    percentage: builtins.float
    rule_kind: builtins.str


# Document
class AnnotatedDocument(BaseModel):
    """Complete annotated document structure"""
    document_id: str
    document_theme: str
    original_document: str
    document_to_annotate: str
    fictionalized_annotated_template_document: str = ""
    rules: List[str] = Field(default_factory=list)
    questions: List[Question] = Field(default_factory=list)
    implicit_rules: List[ImplicitRule] = Field(default_factory=list)
    implicit_rule_exclusions: List[str] = Field(default_factory=list)
    
    @property
    def num_questions(self) -> int:
        """Computed property - returns number of questions"""
        return len(self.questions)


class FictionalDocument(BaseModel):
    """Generated document with fictional entities."""
    document_id: str
    document_theme: str
    generated_document: str
    entities_used: EntityCollection
    questions: List[Dict[str, Any]]
    evaluated_answers: List[Dict[str, Any]]


__all__ = [
    "ENTITY_TYPE_TO_ATTR",
    "ENTITY_TYPE_TO_CLASS",
    "PersonEntity",
    "PlaceEntity",
    "EventEntity",
    "OrganizationEntity",
    "AwardEntity",
    "LegalEntity",
    "ProductEntity",
    "TemporalEntity",
    "NumberEntity",
    "EntityCollection",
    "EntityPool",
    "AnnotationReference",
    "RuleType",
    "Rule",
    "Question",
    "ImplicitRule",
    "AnnotatedDocument",
    "FictionalDocument",
]
