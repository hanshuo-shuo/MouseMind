"""Explicit ownership for a small set of reusable simulator instances.

Callers still reset and verify every rollout. The pool changes construction
cost and resource lifetime only; it does not snapshot or restore simulator
state, and never disables replay verification.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, Callable, Iterator


@contextmanager
def environment_pool(factory: Callable[[], Any], count: int) -> Iterator[tuple[Any, ...]]:
    """Own distinct environments until context exit, including on exceptions."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Environment pool size must be a positive integer")
    with ExitStack() as ownership:
        environments = []
        for _ in range(count):
            environment = factory()
            if any(environment is existing for existing in environments):
                raise ValueError("Environment pool requires independently created instances")
            ownership.callback(environment.close)
            environments.append(environment)
        yield tuple(environments)
