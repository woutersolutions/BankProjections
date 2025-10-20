"""
Logging utilities for adding context and timing to loguru logs.
Provides context managers and iterators that automatically log entry/exit
with timing information and add context to all log messages within scope.
"""

import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from loguru import logger

# Thread-safe context stack using contextvars
_context_stack: ContextVar[list[str]] = ContextVar("_context_stack", default=[])


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
    stack = _context_stack.get().copy()
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
            if timed:
                elapsed = time.time() - start_time
                logger.info(f"Ending {name} in {elapsed:.2f} seconds")

            # Pop context from stack
            stack = _context_stack.get().copy()
            if stack and stack[-1] == name:
                stack.pop()
            _context_stack.set(stack)


def log_iterator(iterable: Iterable[Any], prefix: str = "", suffix: str = "", timed: bool = False) -> Iterator[Any]:
    """
    Iterator wrapper that adds log context for each iteration.
    Each iteration is wrapped in a log context with optional timing and context
    information added to all logs within that iteration.

    Args:
        iterable: The iterable to wrap
        prefix: Base name for the context (default: "")
        timed: If True, logs end message with timing. If False, only logs start message.
    Yields:
        Items from the original iterable
    Example:
        >>> for i in log_iterator(range(3), "processing"):
        ...     logger.info(f"Value: {i}")
        # Logs for each iteration:
        # Starting processing0
        # Value: 0 | processing0 |
        # Ending processing0 in 0.01 seconds

        >>> for i in log_iterator(range(3), "item", timed=False):
        ...     logger.info(f"Value: {i}")
        # Logs for each iteration:
        # Starting item0
        # Value: 0 | item0 |
        # (no ending message)
    """
    for item in iterable:
        if isinstance(item, tuple):
            name = str(item[0])
        else:
            name = str(item)
        context_name = f"{prefix}{name}{suffix}"
        with log_context(context_name, timed=timed):
            yield item


# Configure logger format to include context if present
def setup_logger_format():
    """
    Configure loguru to display context in log messages.
    Call this function to set up the default log format with context support.
    """
    logger.remove()  # Remove default handler
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>{extra[context]}\n",
        colorize=True,
    )


def setup_logger_format_with_context():
    """
    Configure loguru to display context as a prefix in log messages.
    This is the recommended setup for use with log_context and log_iterator.
    """
    logger.remove()  # Remove default handler

    def format_with_context(record):
        """Custom formatter that adds context before the message with colors."""
        context = record["extra"].get("context", "")
        context_prefix = f"<cyan> {context}</cyan>" if context else ""
        return "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> |{context_prefix} | <level>{message}</level>\n".format(
            time=record["time"], level=record["level"].name, message=record["message"], context_prefix=context_prefix
        )

    logger.add(
        lambda msg: print(msg, end=""),
        format=format_with_context,
        colorize=True,
    )
