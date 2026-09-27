"""Generation keeps routine output quiet without hiding failures or leaking state."""

import io
import logging
import sys
import warnings

import pytest

from memoreason.factual_to_fictional_dataset.generation_console import quiet_generation_output


@pytest.fixture
def console_logger():
    logger = logging.getLogger(__name__ + ".generation")
    previous = (logger.level, logger.handlers[:], logger.propagate, logger.disabled)
    previous_disable = logging.root.manager.disable
    log_output = io.StringIO()
    handler = logging.StreamHandler(log_output)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.handlers = [handler]
    logger.propagate = False
    logger.disabled = False
    logger.setLevel(logging.DEBUG)
    logging.disable(logging.NOTSET)
    try:
        yield logger, log_output
    finally:
        logger.setLevel(previous[0])
        logger.handlers = previous[1]
        logger.propagate = previous[2]
        logger.disabled = previous[3]
        handler.close()
        logging.disable(previous_disable)


def test_quiet_generation_hides_routine_output_but_keeps_errors(console_logger, capsys):
    logger, log_output = console_logger
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with quiet_generation_output():
            print("routine progress", flush=True)
            logger.debug("routine debug")
            logger.info("routine info")
            logger.warning("routine logging warning")
            warnings.warn("routine Python warning", UserWarning, stacklevel=2)
            logger.error("generation error")
            logger.critical("critical failure")
            print("explicit error", file=sys.stderr)

    assert recorded == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "explicit error\n"
    assert log_output.getvalue() == "ERROR: generation error\nCRITICAL: critical failure\n"


@pytest.mark.parametrize("fail", [False, True])
def test_quiet_generation_restores_console_state(console_logger, capsys, fail):
    logger, log_output = console_logger
    original_stdout = sys.stdout
    logging.disable(logging.INFO)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        original_filters = warnings.filters[:]

        def generate():
            with quiet_generation_output():
                print("routine progress")
                warnings.warn("routine warning", UserWarning, stacklevel=2)
                if fail:
                    raise RuntimeError("generation failed")

        if fail:
            with pytest.raises(RuntimeError, match="generation failed"):
                generate()
        else:
            generate()

        assert sys.stdout is original_stdout
        assert logging.root.manager.disable == logging.INFO
        assert warnings.filters == original_filters
        print("subsequent output")
        logger.warning("subsequent logging warning")
        warnings.warn("subsequent Python warning", UserWarning, stacklevel=2)

    assert [str(item.message) for item in recorded] == ["subsequent Python warning"]
    captured = capsys.readouterr()
    assert captured.out == "subsequent output\n"
    assert captured.err == ""
    assert log_output.getvalue() == "WARNING: subsequent logging warning\n"


def test_quiet_generation_respects_existing_higher_disable_threshold(console_logger, capsys):
    logger, log_output = console_logger
    logging.disable(logging.CRITICAL)
    with quiet_generation_output():
        logger.error("previously disabled error")
        logger.critical("previously disabled critical")
        assert logging.root.manager.disable == logging.CRITICAL

    assert logging.root.manager.disable == logging.CRITICAL
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert log_output.getvalue() == ""


def test_quiet_generation_preserves_warnings_promoted_to_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="invalid generated data"):
            with quiet_generation_output():
                warnings.warn("invalid generated data", UserWarning, stacklevel=2)
