# Rules Annotation Guide

## Purpose

Rules are safeguards used during fictional document generation when replacing annotated entities.  
They preserve document-specific constraints so replacements stay coherent.

## Annotator task

For each document:

1. Verify drafted rules are valid.
2. Remove invalid or redundant rules.
3. Add missing rules that are clearly required by the text.

## A rule is valid if

- it uses only annotated references from the document
- it is supported by the text (explicitly or directly implied)
- it is executable (`<`, `>`, `<=`, `>=`, `=`, `==`, `!=`, `+`, `-`, `*`, `/`, `century_of(...)`, `century_start(...)`, `century_end(...)`)
- it is useful for coherence after replacement

## Remove rules when

- they only encode generic number/temporal ordering (this is automatically handled by our replacement algorithm)
- they rely on outside knowledge
- they are duplicates

## Add missing rules when

- the text states a clear relation not captured yet (sum, gap, bound, equality)
- breaking that relation would make the fictionalized text inconsistent
- a century mention is explicitly tied to an exact year/date and changing them independently could create a contradiction

## Implicit range rules

- the interface also shows auto-generated implicit range rules for numbers, ages, centuries, and temporals
- these are not AI-drafted semantic rules; they expose the sampling ranges already enforced by the code
- you may edit the lower or upper bound when the default interval is too loose or too tight for the document
- the italic note on each implicit rule states which default convention generated it

## Worked example

Synthetic annotated document:

In [2022; temporal_4.year], [Nora; person_1.name], who was [6; person_1.age] years old, joined the [Oak Street kids' science; entreprise_org_1.name] club. In [2023; temporal_5.year], a year after [she; person_1.subj_pronoun] joined the club, [her; person_1.obj_pronoun] older [brother; person_2.relationship.person_1] [Sam; person_2.name] joined too. [Sam; person_2.name] was [8; person_2.age] years old, so [he; person_1.subj_pronoun] was [2; number_7.int] years older than [Nora; person_1.name]. During the spring tournament, the club won [3; number_2.int] robotics rounds and [2; number_3.int] quiz rounds, for a total of [5; number_4.int] wins. At the summer fair, [Sam; person_2.name] arrived with [4; number_5.int] friends; together with the [3; number_2.int] club children already waiting, their group included [7; number_6.int] children. The club archives say a reform charter was written in the [19th; number_8.int] century and officially adopted in [1875; temporal_6.year]. A related dispute started in the [18th; number_9.int] century and was settled in [1802; temporal_7.year].

## Expected rule examples

For a document like the synthetic example above, good rules capture one explicit constraint that would matter after replacement:

- `number_2.int + number_3.int = number_4.int # number of wins must sum to the total` 
- `temporal_5.year - temporal_4.year == 1 # consecutive calendar years`
- `person_2.age - person_1.age == number_7.int # age gap stated in the document`
- `number_5.int > 1 # plural mention`
- `4 < person_2.age < 12 # age constraints for specific profiles, here for example the text mentions that a person is a child and kids`
- `number_6.int = 2 + number_5.int # the size of the group must be equal to the number of friends plus Nora and Sam`
- `century_of(temporal_6.year) == number_8.int # the adoption year must stay inside the stated century`
- `century_end(number_9.int) < temporal_7.year # the settlement year must come after the stated starting century`

Examples to reject:

- `temporal_4.year < temporal_5.year`
- `number_2.int < number_4.int`
- `person_1.age < person_2.age`

Century note:

- centuries should be annotated as `number` entities, not `temporal`
- only add century rules when the century mention is clearly anchored to an exact year/date in the text

## Writing style

- keep rules minimal
- keep explanations short and factual
