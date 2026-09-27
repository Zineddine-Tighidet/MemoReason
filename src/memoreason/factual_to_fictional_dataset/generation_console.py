"""Minimal terminal output for dataset-generation CLI workers."""

from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
import logging
import os
import warnings

from tqdm import tqdm


@contextmanager
def quiet_generation_output() -> Iterator[None]:
    """Hide routine worker messages without swallowing errors or exceptions."""
    previous_disable = logging.root.manager.disable
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), warnings.catch_warnings():
        # Keep warning filters intact, including warnings promoted to errors.
        warnings.showwarning = lambda *_args, **_kwargs: None
        logging.disable(max(previous_disable, logging.WARNING))
        try:
            yield
        finally:
            logging.disable(previous_disable)


def generation_progress(total: int) -> tqdm:
    """Create the single progress bar owned by the CLI's main process."""
    return tqdm(total=total, desc="Generating", unit="template", dynamic_ncols=True)
