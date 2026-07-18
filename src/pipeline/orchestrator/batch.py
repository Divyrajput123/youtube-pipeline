"""Batch processing helpers for the Orchestrator.

Contains pure functions used by :meth:`~pipeline.orchestrator.Orchestrator.start_batch`
and :meth:`~pipeline.orchestrator.Orchestrator.get_batch_schedule`.

Design reference: §Batch Processing Design (tasks 17.1, 17.2)
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BATCH_MIN = 2
_BATCH_MAX = 10
_SCHEDULE_WINDOW_DAYS = 14


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def validate_batch_size(n: int) -> None:
    """Raise :class:`ValueError` when *n* is outside the allowed range [2, 10].

    Args:
        n: Requested batch size.

    Raises:
        ValueError: When ``n < 2`` or ``n > 10``.
    """
    if not (_BATCH_MIN <= n <= _BATCH_MAX):
        raise ValueError(
            f"Batch size must be between {_BATCH_MIN} and {_BATCH_MAX} inclusive, got {n}."
        )


def generate_batch_slots(n: int, now: datetime) -> list[datetime]:
    """Return *n* evenly-spaced publish datetimes over a 14-day window.

    Formula (design §11.4):
        ``slot[i] = now + i * timedelta(days=14) / (n - 1)``  for ``i in range(n)``

    Special case when ``n == 1``:
        Returns a single slot at ``now + timedelta(days=7)``.

    Args:
        n: Number of slots to generate (must be ≥ 1; the caller should validate
           batch size before calling this function).
        now: Reference datetime used as the base of the scheduling window.

    Returns:
        A list of *n* :class:`datetime` objects in ascending order.

    Examples::

        >>> from datetime import datetime, timezone
        >>> slots = generate_batch_slots(2, datetime(2025, 1, 1, tzinfo=timezone.utc))
        >>> slots[0].isoformat()
        '2025-01-01T00:00:00+00:00'
        >>> slots[1].isoformat()
        '2025-01-15T00:00:00+00:00'
    """
    if n == 1:
        return [now + timedelta(days=7)]

    window = timedelta(days=_SCHEDULE_WINDOW_DAYS)
    return [now + i * window / (n - 1) for i in range(n)]


def compute_batch_completion(published_count: int, total_count: int) -> int:
    """Return the batch completion percentage as a floor integer [0, 100].

    Formula (design §11.6):
        ``floor(published_count / total_count * 100)``

    Returns 0 when *total_count* is 0 (safe division).

    Args:
        published_count: Number of videos that have reached ``Published``.
        total_count: Total number of videos in the batch.

    Returns:
        Integer percentage in [0, 100].

    Examples::

        >>> compute_batch_completion(1, 3)
        33
        >>> compute_batch_completion(3, 3)
        100
        >>> compute_batch_completion(0, 5)
        0
    """
    if total_count == 0:
        return 0
    return math.floor(published_count / total_count * 100)


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "validate_batch_size",
    "generate_batch_slots",
    "compute_batch_completion",
]
