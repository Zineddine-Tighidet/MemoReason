"""Generate remote-model raw outputs in parallel without overwriting artifacts.

The runner supports Groq and Anthropic registry entries.  It can seed a new,
isolated run from prompt-identical raw responses while rebuilding all metadata
against the current immutable dataset.  Existing valid target files are reused;
an existing invalid target is a hard error.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import heapq
import itertools
import json
from pathlib import Path
import threading
import time
from typing import Any

from tqdm import tqdm

from memoreason.factual_to_fictional_dataset.dataset_paths import (
    MODEL_EVAL_RAW_OUTPUTS_DIR,
    model_eval_artifact_path,
)
from memoreason.factual_to_fictional_dataset.dataset_settings import resolve_dataset_settings
from memoreason.model_providers.groq_client import (
    GPT_OSS_REASONING_BUDGET_EXHAUSTED,
    gpt_oss_reasoning_budget_exhaustion,
)
from memoreason.model_providers.text_generation import TextGenerationRequest, generate_text

from .benchmark_document_loading import iter_evaluation_documents
from .model_evaluation_artifact_reuse_contract import (
    _effective_generation_max_tokens,
    _model_generation_context,
)
from .document_question_answering_prompt import (
    DOCUMENT_QA_SYSTEM_PROMPT,
    build_document_question_prompt,
)
from .paper_model_registry import resolve_paper_model_configurations
from .remote_model_answer_artifacts import (
    build_raw_document_payload,
    existing_target_is_current,
    matching_seed_path,
    seed_results_for_document,
    write_json_exclusive,
    write_yaml_exclusive,
)


_TaskLabel = tuple[int, int]


def _new_result(
    *,
    model_configuration: object,
    document: object,
    question: object,
) -> dict[str, Any]:
    user_prompt = build_document_question_prompt(
        document.document_text,
        question.question_text,
        answer_schema=question.answer_schema,
    )
    effective_max_tokens = _effective_generation_max_tokens(model_configuration, question)
    response = generate_text(
        TextGenerationRequest(
            provider=model_configuration.provider,
            model=model_configuration.model_name,
            system_prompt=DOCUMENT_QA_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=model_configuration.temperature,
            max_tokens=effective_max_tokens,
            seed=model_configuration.seed,
        )
    )
    special_outcome: dict[str, object] = {}
    if not response.text.strip():
        try:
            raw_provider_response = json.loads(response.raw_response)
        except json.JSONDecodeError:
            raw_provider_response = None
        termination = (
            gpt_oss_reasoning_budget_exhaustion(
                raw_provider_response,
                max_completion_tokens=effective_max_tokens,
            )
            if model_configuration.provider == "groq"
            else None
        )
        if termination is None:
            raise RuntimeError(
                f"Provider returned an unproven empty answer for "
                f"{model_configuration.model_id}/{document.document_id}/"
                f"{document.document_setting}/{document.document_variant_id}/"
                f"{question.question_id}"
            )
        special_outcome = {
            "generation_outcome": GPT_OSS_REASONING_BUDGET_EXHAUSTED,
            "generation_termination": termination,
        }
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "answer_behavior": question.answer_behavior,
        "question_text": question.question_text,
        "ground_truth": question.ground_truth,
        "ground_truth_canonical": question.ground_truth_canonical,
        "answer_schema": question.answer_schema,
        "answer_expression": question.answer_expression,
        "accepted_answer_overrides": list(question.accepted_answer_overrides),
        "accepted_answers": list(question.accepted_answers),
        "accepted_answers_canonical": list(question.accepted_answers_canonical),
        "pair_key": question.pair_key,
        "user_prompt": user_prompt,
        "effective_max_tokens": effective_max_tokens,
        "raw_output": response.text,
        "raw_reasoning": response.reasoning_text,
        "raw_provider_response": response.raw_response,
        **special_outcome,
    }


def _task_retry_delay_seconds(failed_attempts: int) -> float:
    return min(5.0, 0.25 * failed_attempts)


def _run_tasks_with_scheduled_retries(
    *,
    labels: Iterable[_TaskLabel],
    workers: int,
    max_attempts: int,
    submit: Callable[[ThreadPoolExecutor, _TaskLabel], Future[dict[str, Any]]],
    on_success: Callable[[_TaskLabel, dict[str, Any]], None],
    failure_context: Callable[[_TaskLabel], str],
    retry_delay_seconds: Callable[[int], float] = _task_retry_delay_seconds,
) -> None:
    """Run tasks while scheduling retries without serially sleeping per failure.

    A provider outage can complete thousands of futures with errors at nearly
    the same time. Sleeping while processing each failed future makes the
    consumer accumulate every backoff sequentially and prevents it from
    draining already-completed work. Failed labels are instead placed on a
    stable deadline heap. The consumer keeps draining live futures and only
    waits once, until the earliest retry deadline, when no future is in flight.
    """

    worker_count = max(1, int(workers))
    attempt_limit = max(1, int(max_attempts))
    failed_attempts: defaultdict[_TaskLabel, int] = defaultdict(int)
    future_labels: dict[Future[dict[str, Any]], _TaskLabel] = {}
    scheduled_retries: list[tuple[float, int, _TaskLabel]] = []
    retry_order = itertools.count()
    retry_wait = threading.Event()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:

        def _submit(label: _TaskLabel) -> None:
            future = submit(executor, label)
            future_labels[future] = label

        for label in labels:
            _submit(label)

        while future_labels or scheduled_retries:
            now = time.monotonic()
            while scheduled_retries and scheduled_retries[0][0] <= now:
                _, _, label = heapq.heappop(scheduled_retries)
                _submit(label)

            if not future_labels:
                retry_wait.wait(max(0.0, scheduled_retries[0][0] - time.monotonic()))
                continue

            timeout = None
            if scheduled_retries:
                timeout = max(0.0, scheduled_retries[0][0] - time.monotonic())
            done, _ = wait(future_labels, timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                continue

            # ``done`` is a set. Sorting by the immutable dataset position
            # makes simultaneous failures enter the retry heap deterministically.
            for future in sorted(done, key=future_labels.__getitem__):
                label = future_labels.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    failed_attempts[label] += 1
                    if failed_attempts[label] >= attempt_limit:
                        raise RuntimeError(f"{failure_context(label)}: {exc}") from exc
                    delay = max(0.0, float(retry_delay_seconds(failed_attempts[label])))
                    heapq.heappush(
                        scheduled_retries,
                        (time.monotonic() + delay, next(retry_order), label),
                    )
                    continue
                on_success(label, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="One Groq or Anthropic registry name.")
    parser.add_argument("--settings", nargs="+", required=True)
    parser.add_argument("--source-raw-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--task-max-attempts", type=int, default=20)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--expected-network-requests", type=int, default=None)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.summary_json.exists() or args.summary_json.is_symlink():
        raise FileExistsError(f"Refusing to overwrite summary: {args.summary_json}")
    if not args.plan_only and args.expected_network_requests is None:
        parser.error("--expected-network-requests is required for a network run")
    source_raw_root = args.source_raw_root.resolve(strict=True) if args.source_raw_root else None

    model_configuration = resolve_paper_model_configurations([args.model])[0]
    if model_configuration.provider not in {"groq", "anthropic"}:
        raise ValueError(
            f"{model_configuration.model_id!r} uses provider={model_configuration.provider!r}; only Groq and Anthropic are allowed"
        )
    setting_ids = [
        spec.setting_id
        for spec in resolve_dataset_settings(
            explicit_settings=args.settings,
            include_factual=True,
        )
    ]
    documents = list(iter_evaluation_documents(settings=setting_ids))
    counts: Counter[str] = Counter()
    jobs: list[dict[str, Any]] = []

    for document in documents:
        output_path = model_eval_artifact_path(
            theme=document.document_theme,
            model_name=model_configuration.model_id,
            document_id=document.document_id,
            setting=document.document_setting,
            stage_suffix="raw_outputs",
            variant_id=(
                None
                if document.document_variant_index == 1 and document.source_path.stem == document.document_id
                else document.document_variant_id
            ),
        )
        counts["expected_documents"] += 1
        counts["expected_rows"] += len(document.questions)
        if output_path.exists():
            if not existing_target_is_current(
                output_path,
                model_configuration=model_configuration,
                document=document,
            ):
                raise RuntimeError(f"Refusing to overwrite an existing invalid target: {output_path}")
            counts["existing_documents"] += 1
            counts["existing_rows"] += len(document.questions)
            continue

        reusable = seed_results_for_document(
            matching_seed_path(source_raw_root, output_path),
            model_configuration=model_configuration,
            document=document,
        )
        results: list[dict[str, Any] | None] = [reusable.get(question.question_id) for question in document.questions]
        question_indices = [index for index, result in enumerate(results) if result is None]
        counts["new_documents"] += 1
        counts["seed_reused_rows"] += len(document.questions) - len(question_indices)
        counts["network_required_rows"] += len(question_indices)
        jobs.append(
            {
                "model_configuration": model_configuration,
                "document": document,
                "output_path": output_path,
                "results": results,
                "question_indices": question_indices,
                "remaining": len(question_indices),
            }
        )

    plan = {
        "schema_version": 1,
        "mode": "plan" if args.plan_only else "run",
        "model": model_configuration.model_id,
        "provider": model_configuration.provider,
        "provider_model_name": model_configuration.model_name,
        "settings": setting_ids,
        "source_raw_root": str(source_raw_root) if source_raw_root else None,
        "target_raw_root": str(MODEL_EVAL_RAW_OUTPUTS_DIR),
        "counts": dict(sorted(counts.items())),
        "generation_config": _model_generation_context(model_configuration),
    }
    if args.plan_only:
        write_json_exclusive(plan, args.summary_json)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    expected_network_requests = int(args.expected_network_requests)
    actual_network_requests = int(counts["network_required_rows"])
    if actual_network_requests != expected_network_requests:
        raise RuntimeError(
            f"Network-request guard failed: expected={expected_network_requests} actual={actual_network_requests}"
        )

    saved_documents = 0
    completed_network_requests = 0
    lock = threading.Lock()

    def _save_if_complete(job_index: int) -> bool:
        nonlocal saved_documents
        job = jobs[job_index]
        if job["remaining"] != 0:
            return False
        results = job["results"]
        if any(result is None for result in results):
            raise RuntimeError(f"Incomplete result vector for {job['output_path']}")
        write_yaml_exclusive(
            build_raw_document_payload(
                model_configuration=job["model_configuration"],
                document=job["document"],
                results=list(results),
            ),
            job["output_path"],
        )
        saved_documents += 1
        return True

    for job_index, job in enumerate(jobs):
        if job["remaining"] == 0:
            _save_if_complete(job_index)

    workers = max(1, int(args.workers))
    max_attempts = max(1, int(args.task_max_attempts))
    labels = (
        (job_index, question_index) for job_index, job in enumerate(jobs) for question_index in job["question_indices"]
    )

    def _submit(executor: ThreadPoolExecutor, label: _TaskLabel) -> Future[dict[str, Any]]:
        job_index, question_index = label
        job = jobs[job_index]
        return executor.submit(
            _new_result,
            model_configuration=job["model_configuration"],
            document=job["document"],
            question=job["document"].questions[question_index],
        )

    def _failure_context(label: _TaskLabel) -> str:
        job_index, question_index = label
        job = jobs[job_index]
        question = job["document"].questions[question_index]
        return (
            f"Question failed after {max_attempts} attempts for "
            f"{model_configuration.model_id}/{job['document'].document_id}/{question.question_id}"
        )

    with tqdm(
        total=actual_network_requests,
        unit="question",
        desc=f"Raw {model_configuration.provider} eval",
        dynamic_ncols=True,
    ) as progress:

        def _on_success(label: _TaskLabel, result: dict[str, Any]) -> None:
            nonlocal completed_network_requests
            job_index, question_index = label
            with lock:
                job = jobs[job_index]
                job["results"][question_index] = result
                job["remaining"] -= 1
                completed_network_requests += 1
                saved_now = _save_if_complete(job_index)
                progress.set_postfix(
                    saved_documents=saved_documents,
                    last_saved=int(saved_now),
                )
                progress.update(1)

        _run_tasks_with_scheduled_retries(
            labels=labels,
            workers=workers,
            max_attempts=max_attempts,
            submit=_submit,
            on_success=_on_success,
            failure_context=_failure_context,
        )

    result_summary = {
        **plan,
        "mode": "complete",
        "completed_network_requests": completed_network_requests,
        "saved_documents": saved_documents,
    }
    write_json_exclusive(result_summary, args.summary_json)
    print(json.dumps(result_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
