"""
Logging utilities for adding context and timing to loguru logs.
Provides context managers and iterators that automatically log entry/exit
with timing information and add context to all log messages within scope.
"""

import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar

from loguru import logger

T = TypeVar("T")

# Thread-safe context stack using contextvars
_context_stack: ContextVar[list[str] | None] = ContextVar("_context_stack", default=None)


@contextmanager
def log_context(name: str, timed: bool = True) -> Iterator[None]:
    """
    Context manager that adds context to log messages and optionally times execution.
    Logs entry and (if timed) exit messages with timing information, and adds the context
    name to all log messages within the context block.
    Supports nested contexts which will be displayed as: parent | child | grandchild

    Args:
        name: The context name to use in log messages
        timed: If True, logs end message with timing. If False, only logs start message.
    Example:
        >>> with log_context("data_processing"):
        ...     logger.info("Processing data")
        # Logs:
        # Starting data_processing
        # Processing data | data_processing |
        # Ending data_processing in 0.05 seconds

        >>> with log_context("quick_task", timed=False):
        ...     logger.info("Doing task")
        # Logs:
        # Starting quick_task
        # Doing task | quick_task |
        # (no ending message)
    """
    start_time = time.time() if timed else None

    # Get current stack and add new context
    stack = (_context_stack.get() or []).copy()
    stack.append(name)
    _context_stack.set(stack)

    # Create context string from stack
    context_str = " | ".join(stack)

    # Add context to all logs within this block
    with logger.contextualize(context=context_str):
        logger.info(f"Starting {name}")

        try:
            yield
        finally:
            if timed and start_time is not None:
                elapsed = time.time() - start_time
                logger.info(f"Ending {name} in {elapsed:.2f} seconds")

            # Pop context from stack
            stack = (_context_stack.get() or []).copy()
            if stack and stack[-1] == name:
                stack.pop()
            _context_stack.set(stack)


def log_iterator(
    iterable: Iterable[T],
    prefix: str = "",
    suffix: str = "",
    timed: bool = False,
    show_progress: bool = True,
    item_name: Callable[[T], str] | None = str,
) -> Iterator[T]:
    """
    Iterator wrapper that adds log context for each iteration.

    Args:
        iterable: The iterable to wrap
        prefix: Prefix for the context name (default: "")
        suffix: Suffix appended after progress (default: "")
        timed: If True, logs end message with timing
        show_progress: If True, shows "1/N" style progress (requires sized iterable)
        item_name: Optional callable to extract display name from item.
                   If None, uses 1-based index for progress display.

    Yields:
        Items from the original iterable

    Example:
        >>> for item in log_iterator(items, prefix="Processing ", timed=True):
        ...     logger.info(f"Working on {item}")
        # Logs: "Starting Processing 1/3", "Starting Processing 2/3", etc.

        >>> for key, value in log_iterator(mapping.items(), prefix="Key ", item_name=lambda x: x[0]):
        ...     process(value)
        # Logs: "Starting Key foo 1/3", "Starting Key bar 2/3", etc.
    """
    # Convert to list if we need length for progress display
    if show_progress:
        items = list(iterable)
        total = len(items)
    else:
        items = iterable
        total = None

    for i, item in enumerate(items, 1):
        # Build context name
        name_part = item_name(item) if item_name is not None else str(i)

        if show_progress and total is not None:
            progress_part = f"{i}/{total}"
            # If using item_name, include both name and progress
            if item_name is not None:
                context_name = f"{prefix}{name_part} {progress_part}{suffix}"
            else:
                context_name = f"{prefix}{progress_part}{suffix}"
        else:
            context_name = f"{prefix}{name_part}{suffix}"

        with log_context(context_name, timed=timed):
            yield item


# Configure logger format to include context if present
def setup_logger_format() -> None:
    """
    Configure loguru to display context in log messages.
    Call this function to set up the default log format with context support.
    """
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<level>{message}</level>{extra[context]}\n",
        colorize=True,
    )


def setup_logger_format_with_context() -> None:
    """
    Configure loguru to display context as a prefix in log messages.
    This is the recommended setup for use with log_context and log_iterator.
    """
    logger.remove()  # Remove default handler

    def format_with_context(record: dict[str, Any]) -> str:
        """Custom formatter that adds context before the message with colors."""
        context = record["extra"].get("context", "")
        context_prefix = f"<cyan> {context}</cyan>" if context else ""
        template = [
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green>",
            "<level>{level: <8}</level>",
            "{context_prefix}",
            "<level>{message}</level>\n",
        ]
        return " | ".join(template).format(
            time=record["time"], level=record["level"].name, message=record["message"], context_prefix=context_prefix
        )

    logger.add(
        lambda msg: print(msg, end=""),
        format=format_with_context,  # type: ignore[arg-type]
        colorize=True,
    )
