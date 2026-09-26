"""Temporal generation workflow driven by randomized constrained sampling."""

from __future__ import annotations

from datetime import date
import random

from memoreason.benchmark_definition.document_schema import EntityCollection, TemporalEntity

from ..generation_limits import _MIN_YEAR
from .shift_fallback import TemporalShiftFallbackMixin


class TemporalGenerationWorkflowMixin(TemporalShiftFallbackMixin):
    """Generate temporal entities with randomized year sampling inside local windows."""

    def _sample_temporal_years_randomly(
        self,
        required_temporals: list[tuple],
        rules: list[str],
        existing_entities: EntityCollection | None,
        excluded_years: set[int],
        decade_year_temporal_ids: set[str],
        *,
        max_attempts: int = 256,
    ) -> dict[str, int] | None:
        """Sample temporal years with randomized constraint-aware assignments.

        This mirrors the paper-level generation story more closely than the
        older temporal MILP path: years are sampled from bounded local domains,
        then accepted only when they remain compatible with the explicit rules
        and with factual chronology constraints.
        """
        required_year_ids = [
            temporal_id for temporal_id, attrs in required_temporals if any(attr in attrs for attr in ("year", "date"))
        ]
        if not required_year_ids:
            return {}

        constraints = self._collect_temporal_year_constraints(required_temporals, rules, existing_entities)
        if constraints is None:
            return None
        constraints.extend(self._ordering_temporal_year_constraints(required_temporals, existing_entities))
        if not constraints:
            return None

        domains: dict[str, list[int]] = {}
        for temporal_id in required_year_ids:
            base_lo, base_hi = self._temporal_year_base_range(temporal_id)
            domain = self._temporal_year_domain(
                temporal_id,
                base_lo,
                base_hi,
                excluded_years,
                decade_year_temporal_ids,
            )
            if not domain:
                _, expanded_max_year = self._temporal_year_sampling_bounds(temporal_id)
                domain = self._expand_temporal_year_domain(
                    temporal_id,
                    _MIN_YEAR,
                    expanded_max_year,
                    excluded_years,
                    decade_year_temporal_ids,
                )
            if not domain:
                return None
            domains[temporal_id] = domain

        domain_bounds = {temporal_id: (min(domain), max(domain)) for temporal_id, domain in domains.items()}
        sampling_order = sorted(required_year_ids, key=lambda temporal_id: (len(domains[temporal_id]), temporal_id))

        for _ in range(max_attempts):
            assignments: dict[str, int] = {}
            failed = False
            for temporal_id in sampling_order:
                candidate_years = list(domains[temporal_id])
                random.shuffle(candidate_years)
                valid_years: list[int] = []
                for candidate_year in candidate_years:
                    assignments[temporal_id] = candidate_year
                    if self._constraints_feasible(constraints, assignments, domain_bounds):
                        valid_years.append(candidate_year)
                    assignments.pop(temporal_id, None)
                if not valid_years:
                    failed = True
                    break
                assignments[temporal_id] = random.choice(valid_years)
            if failed:
                continue
            exact_domains = {
                temporal_id: (assigned_year, assigned_year) for temporal_id, assigned_year in assignments.items()
            }
            if self._constraints_feasible(constraints, assignments, exact_domains):
                return assignments

        return None

    def generate_temporals_with_rules(
        self,
        required_temporals: list[tuple],
        rules: list[str] | None,
        existing_entities: EntityCollection | None,
        decade_year_temporal_ids: set[str] | None = None,
    ) -> dict[str, TemporalEntity]:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        months = list(self._MONTHS)
        excluded_days = self.exclude_temporals.get("days", set())
        excluded_months = self.exclude_temporals.get("months", set())
        excluded_years = self.exclude_temporals.get("years", set())
        excluded_day_of_months = self.exclude_temporals.get("day_of_months", set())
        excluded_days_by_id = self.exclude_temporals.get("days_by_id", {})
        excluded_months_by_id = self.exclude_temporals.get("months_by_id", {})
        excluded_day_of_months_by_id = self.exclude_temporals.get("day_of_months_by_id", {})
        excluded_timestamps_by_id = self.exclude_temporals.get("timestamps_by_id", {})
        decade_year_temporal_ids = set(decade_year_temporal_ids or set())

        def choose_available(candidates: list, global_excluded: set, by_id_excluded: dict, temporal_id: str):
            # Values factual for sibling refs are a soft preference only.  A
            # ref's own factual value and its prior-variant values live in the
            # per-ID set and remain hard exclusions in both passes.
            per_id_excluded = set(by_id_excluded.get(temporal_id) or set())
            available = [
                candidate
                for candidate in candidates
                if candidate not in global_excluded and candidate not in per_id_excluded
            ]
            if available:
                return random.choice(available)
            fallback = [candidate for candidate in candidates if candidate not in per_id_excluded]
            if fallback:
                return random.choice(fallback)
            return None

        temporal_rules = rules or []
        has_explicit_temporal_rules = bool(temporal_rules)
        sampled_years = None
        exact_solve_certified_infeasible = False
        if has_explicit_temporal_rules:
            # Prefer the exact temporal solver first when explicit year rules exist.
            # The randomized sampler is a good fallback, but dense templates with
            # cross-linked year equations can spend a long time exploring domains.
            exact_result = self._solve_temporal_years_with_status(
                required_temporals,
                temporal_rules,
                existing_entities,
                excluded_years,
                decade_year_temporal_ids,
            )
            sampled_years = exact_result.assignments
            exact_solve_certified_infeasible = exact_result.is_certified_infeasible
        if sampled_years is None and has_explicit_temporal_rules and not exact_solve_certified_infeasible:
            sampled_years = self._sample_temporal_years_randomly(
                required_temporals,
                temporal_rules,
                existing_entities,
                excluded_years,
                decade_year_temporal_ids,
            )
        if sampled_years is not None and not self._temporal_assignments_satisfy_constraints(
            sampled_years,
            required_temporals,
            temporal_rules,
            existing_entities,
        ):
            sampled_years = None
        if sampled_years is None and has_explicit_temporal_rules:
            shifted_years = self._generate_shifted_required_years(
                required_temporals,
                existing_entities,
                excluded_years,
                decade_year_temporal_ids,
            )
            if shifted_years is not None and self._temporal_assignments_satisfy_constraints(
                shifted_years,
                required_temporals,
                temporal_rules,
                existing_entities,
            ):
                sampled_years = shifted_years

        # Pre-compute ordering-aware years if no explicit temporal constraints solved.
        ordered_years: dict[str, int] = {}
        if sampled_years is None:
            ordered_years = self._generate_ordered_years(
                required_temporals,
                excluded_years,
                decade_year_temporal_ids,
            )
            if ordered_years is None or (
                has_explicit_temporal_rules
                and not self._temporal_assignments_satisfy_constraints(
                    ordered_years,
                    required_temporals,
                    temporal_rules,
                    existing_entities,
                )
            ):
                raise ValueError("Unable to generate temporals while preserving factual chronological ordering.")

        temporals = {}
        for temporal_id, attrs in required_temporals:
            entity = TemporalEntity()
            if "timestamp" in attrs:
                factual_timestamp = self._factual_temporal_timestamp(temporal_id)
                if factual_timestamp:
                    used_timestamps = {
                        str(value) for value in (excluded_timestamps_by_id.get(temporal_id) or set()) if value
                    }
                    candidate_count = max(8, len(used_timestamps) + 1)
                    for _ in range(candidate_count):
                        candidate_timestamp = self._fictionalize_timestamp_surface(factual_timestamp)
                        entity.timestamp = candidate_timestamp
                        if candidate_timestamp not in used_timestamps:
                            break
            if "day" in attrs and "date" not in attrs:
                entity.day = choose_available(days, excluded_days, excluded_days_by_id, temporal_id)
            if "day_of_month" in attrs:
                entity.day_of_month = choose_available(
                    list(range(1, 29)),
                    excluded_day_of_months,
                    excluded_day_of_months_by_id,
                    temporal_id,
                )
            if "month" in attrs:
                entity.month = choose_available(months, excluded_months, excluded_months_by_id, temporal_id)
            if "year" in attrs or "date" in attrs or (sampled_years and temporal_id in sampled_years):
                requires_decade_year = temporal_id in decade_year_temporal_ids
                if sampled_years is not None and temporal_id in sampled_years:
                    sampled_year = sampled_years[temporal_id]
                    if (not requires_decade_year) or (sampled_year % 10 == 0):
                        entity.year = sampled_year
                elif temporal_id in ordered_years:
                    ordered_year = ordered_years[temporal_id]
                    if (not requires_decade_year) or (ordered_year % 10 == 0):
                        entity.year = ordered_year
                if entity.year is None:
                    base_lo, base_hi = self._temporal_year_base_range(temporal_id)
                    available_years = self._temporal_year_domain(
                        temporal_id,
                        base_lo,
                        base_hi,
                        excluded_years,
                        decade_year_temporal_ids,
                    )
                    if not available_years:
                        _, expanded_max_year = self._temporal_year_sampling_bounds(temporal_id)
                        available_years = self._temporal_year_domain(
                            temporal_id,
                            _MIN_YEAR,
                            expanded_max_year,
                            excluded_years,
                            decade_year_temporal_ids,
                        )
                    if available_years:
                        entity.year = random.choice(available_years)
                    else:
                        raise ValueError(f"No valid year candidates remain for {temporal_id}.")
            if "date" in attrs:
                if not entity.month:
                    entity.month = choose_available(months, excluded_months, excluded_months_by_id, temporal_id)
                if entity.day_of_month is not None:
                    day_num = int(entity.day_of_month)
                else:
                    day_num = int(
                        choose_available(
                            list(range(1, 29)),
                            excluded_day_of_months,
                            excluded_day_of_months_by_id,
                            temporal_id,
                        )
                    )
                    entity.day_of_month = day_num
                if "day" in attrs and entity.month and entity.year:
                    month_number = self._month_name_to_number(entity.month)
                    if month_number is None:
                        raise RuntimeError(
                            f"Temporal generation produced an unknown month for {temporal_id}: {entity.month!r}."
                        )
                    per_id_excluded_days = set(excluded_days_by_id.get(temporal_id) or set())
                    per_id_excluded_dates = set(excluded_day_of_months_by_id.get(temporal_id) or set())
                    allowed_day_numbers = [
                        candidate
                        for candidate in range(1, 29)
                        if candidate not in excluded_day_of_months and candidate not in per_id_excluded_dates
                    ]
                    if not allowed_day_numbers:
                        allowed_day_numbers = [
                            candidate for candidate in range(1, 29) if candidate not in excluded_day_of_months
                        ]
                    if not allowed_day_numbers:
                        allowed_day_numbers = list(range(1, 29))

                    factual_day = None
                    if self.factual_entities is not None:
                        factual_temporal = self.factual_entities.temporals.get(temporal_id)
                        factual_day = str(getattr(factual_temporal, "day", "") or "").strip() or None

                    weekday_preferences = (
                        set(excluded_days) | per_id_excluded_days,
                        ({factual_day} if factual_day else set()) | per_id_excluded_days,
                        {factual_day} if factual_day else set(),
                        set(),
                    )
                    weekday_by_day_number = {
                        candidate: date(int(entity.year), month_number, candidate).strftime("%A")
                        for candidate in allowed_day_numbers
                    }
                    for disallowed_weekdays in weekday_preferences:
                        weekday_compatible_days = [
                            candidate
                            for candidate in allowed_day_numbers
                            if weekday_by_day_number[candidate] not in disallowed_weekdays
                        ]
                        if weekday_compatible_days:
                            day_num = random.choice(weekday_compatible_days)
                            entity.day_of_month = day_num
                            break
                if entity.month and entity.year:
                    entity.date = f"{day_num} {entity.month} {entity.year}"
                    if "day" in attrs:
                        generated_date = self._temporal_entity_to_date(entity)
                        if generated_date is None:
                            raise RuntimeError(
                                f"Temporal generation could not parse its own date for {temporal_id}: {entity.date!r}."
                            )
                        entity.day = generated_date.strftime("%A")
                elif entity.year:
                    raise RuntimeError(
                        f"Temporal generation missing month for {temporal_id} while rendering date "
                        f"(year={entity.year}, day_of_month={day_num})."
                    )
            temporals[temporal_id] = entity
        self._apply_date_difference_rules(temporals, temporal_rules, existing_entities)
        return temporals

    def _generate_ordered_years(
        self,
        required_temporals: list[tuple],
        excluded_years: set[int],
        decade_year_temporal_ids: set[str],
    ) -> dict[str, int] | None:
        """Generate fictional years that preserve factual chronological order (not gaps).

        This samples *ordered* years independently so temporal gaps can change.
        Gaps may still coincidentally match factual values due to randomness,
        which is acceptable. Ordering is preserved relative to any kept
        temporals and between replaced temporals that have factual years.

        Returns a mapping of temporal_id -> fictional_year for every temporal
        in ``required_temporals`` that has a factual year reference.
        """
        if not self.factual_entities or not self.factual_entities.temporals:
            return {}

        replacing_ids: set[str] = set()
        for temporal_id, attrs in required_temporals:
            if "year" in attrs or "date" in attrs:
                replacing_ids.add(temporal_id)

        if not replacing_ids:
            return {}

        all_factual: dict[str, int] = {}
        for tid, factual_entity in self.factual_entities.temporals.items():
            factual_year = getattr(factual_entity, "year", None)
            if factual_year is not None:
                all_factual[tid] = factual_year

        replaced_with_years = {tid: all_factual[tid] for tid in replacing_ids if tid in all_factual}
        if not replaced_with_years:
            return {}

        kept_with_years = {tid: yr for tid, yr in all_factual.items() if tid not in replacing_ids}

        # Determine per-temporal bounds based on kept temporals.
        bounds: dict[str, tuple[int, int]] = {}
        for tid, factual_year in replaced_with_years.items():
            lo, hi = self._temporal_year_base_range(tid)
            for _, kept_year in kept_with_years.items():
                if kept_year < factual_year:
                    lo = max(lo, kept_year + 1)
                elif kept_year > factual_year:
                    hi = min(hi, kept_year - 1)
            bounds[tid] = (lo, hi)

        ordered = sorted(replaced_with_years.items(), key=lambda kv: (kv[1], kv[0]))
        ids = [tid for tid, _ in ordered]
        factual_years = [yr for _, yr in ordered]

        # Compute earliest / latest feasible years to preserve ordering.
        earliest: list[int] = []
        latest: list[int] = []
        for i, tid in enumerate(ids):
            lo, _ = bounds[tid]
            if i == 0:
                earliest.append(lo)
            else:
                delta = 1 if factual_years[i - 1] < factual_years[i] else 0
                earliest.append(max(lo, earliest[i - 1] + delta))
        for i in reversed(range(len(ids))):
            _, hi = bounds[ids[i]]
            if i == len(ids) - 1:
                latest.append(hi)
            else:
                delta = 1 if factual_years[i] < factual_years[i + 1] else 0
                latest_val = min(hi, latest[-1] - delta)
                latest.append(latest_val)
        latest = list(reversed(latest))

        if any(earliest_value > latest_value for earliest_value, latest_value in zip(earliest, latest, strict=False)):
            return None

        candidate_domains: list[list[int]] = []
        for i, tid in enumerate(ids):
            candidates = self._temporal_year_domain(
                tid,
                earliest[i],
                latest[i],
                excluded_years,
                decade_year_temporal_ids,
            )
            if not candidates:
                candidates = list(range(earliest[i], latest[i] + 1))
                if tid in decade_year_temporal_ids:
                    candidates = [year for year in candidates if year % 10 == 0]
            if not candidates:
                return None
            candidate_domains.append(candidates)

        # Refine continuous earliest/latest bounds against sparse domains
        # (notably decade-valued years).  Greedy forward sampling can otherwise
        # choose an early entity too late and strand a later entity whose only
        # feasible value is, for example, 2020.
        feasible_domains: list[list[int]] = [[] for _ in ids]
        for i in reversed(range(len(ids))):
            candidates = candidate_domains[i]
            if i < len(ids) - 1:
                delta = 1 if factual_years[i] < factual_years[i + 1] else 0
                latest_for_suffix = max(feasible_domains[i + 1]) - delta
                candidates = [year for year in candidates if year <= latest_for_suffix]
            if not candidates:
                return None
            feasible_domains[i] = candidates

        result: dict[str, int] = {}
        prev_year = None
        for i, tid in enumerate(ids):
            delta = 1 if i > 0 and factual_years[i - 1] < factual_years[i] else 0
            min_allowed = (prev_year + delta) if prev_year is not None else earliest[i]
            candidates = [year for year in feasible_domains[i] if year >= min_allowed]
            if not candidates:
                return None
            chosen = random.choice(candidates)
            result[tid] = chosen
            prev_year = chosen

        return result
