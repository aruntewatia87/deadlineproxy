"""Deadlines and request-budget allocation.

A :class:`Deadline` is an absolute point in time by which a request must be
finished.  The single most important property of a deadline is that it is
**shared**: every attempt of a request, and every sub-operation within an
attempt, must be derived from the same deadline rather than each getting its
own fresh timeout.  A gateway that gives each retry a full timeout can
spend ``max_attempts x timeout`` on a request that was supposed to finish in
``timeout`` -- that is how a 137 second hang happens when the client asked
for 30.

The :class:`DeadlineBudget` models the way a real proxy must split a single
end-to-end budget between the stages of a request: some time to pick a
route, some to call the provider, some to validate the response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class DeadlineExceeded(Exception):
    """Raised when an operation cannot complete within its deadline."""


class Deadline:
    """An absolute point in time by which work must complete.

    The deadline is expressed in *monotonic* seconds so that a wall-clock
    adjustment cannot make it appear to move.
    """

    __slots__ = ("_end",)

    def __init__(self, end: float) -> None:
        """Create a deadline that expires at the given monotonic timestamp.

        Args:
            end: The monotonic timestamp at which the deadline expires.
        """
        self._end = float(end)

    @classmethod
    def from_timeout(cls, seconds: float, *, clock: object = time.monotonic) -> Deadline:
        """Create a deadline ``seconds`` from now.

        Args:
            seconds: The time budget.
            clock: Monotonic clock callable, injectable for tests.

        Returns:
            The new deadline.

        Raises:
            ValueError: If ``seconds`` is negative.
        """
        if seconds < 0:
            raise ValueError(f"timeout must be non-negative, got {seconds}")
        return cls(clock() + float(seconds))  # type: ignore[operator]

    @classmethod
    def at(cls, end: float) -> Deadline:
        """Create a deadline that expires at an absolute monotonic timestamp.

        Args:
            end: The monotonic timestamp at which the deadline expires.
        """
        return cls(end)

    def remaining(self, *, clock: object = time.monotonic) -> float:
        """Seconds left before the deadline expires (never negative)."""
        return max(0.0, self._end - clock())  # type: ignore[operator]

    def expired(self, *, clock: object = time.monotonic) -> bool:
        """Whether the deadline has already passed."""
        return self.remaining(clock=clock) <= 0.0

    def with_timeout(self, seconds: float, *, clock: object = time.monotonic) -> Deadline:
        """Return a deadline that is at most ``seconds`` from now.

        The result is the earlier of the current deadline and the new one, so
        narrowing a deadline can never extend it.

        Args:
            seconds: The additional time budget.
            clock: Monotonic clock callable, injectable for tests.

        Returns:
            The narrowed deadline.
        """
        if seconds < 0:
            raise ValueError(f"timeout must be non-negative, got {seconds}")
        return Deadline(min(self._end, clock() + float(seconds)))  # type: ignore[operator]

    def child(self, share: float, *, clock: object = time.monotonic) -> Deadline:
        """Derive a sub-deadline for a nested operation.

        The child gets ``share`` of the *remaining* time, so the deadline
        genuinely shortens: a nested operation that is allowed half the
        remaining time must finish in half the remaining time, not in the
        whole of it.

        Args:
            share: The fraction of the remaining budget to give the child.
            clock: Monotonic clock callable, injectable for tests.

        Returns:
            The child deadline.

        Raises:
            ValueError: If ``share`` is not in ``(0, 1]``.
        """
        if not 0.0 < share <= 1.0:
            raise ValueError(f"share must be in (0, 1], got {share}")
        remaining = self.remaining(clock=clock)
        return Deadline(self._end - remaining * (1.0 - share))

    def __repr__(self) -> str:
        return f"Deadline(remaining={self.remaining():.3f}s)"


@dataclass(slots=True)
class DeadlineBudget:
    """A budget of time split between named stages of a request.

    Example
    -------
    >>> budget = DeadlineBudget(total=1.0)
    >>> budget.reserve("routing", 0.2)
    0.2
    >>> budget.reserve("provider", 0.7)
    0.7
    """

    total: float
    reserved: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError(f"total must be non-negative, got {self.total}")

    def reserve(self, name: str, seconds: float) -> float:
        """Reserve ``seconds`` of the budget for the stage called ``name``.

        Args:
            name: The stage name.
            seconds: The amount of time to reserve.

        Returns:
            The amount actually reserved, which is the requested amount
            clamped to what is left.

        Raises:
            ValueError: If ``seconds`` is negative.
        """
        if seconds < 0:
            raise ValueError(f"seconds must be non-negative, got {seconds}")
        available = max(0.0, self.total - self.used)
        granted = min(float(seconds), available)
        self.reserved[name] = self.reserved.get(name, 0.0) + granted
        return granted

    @property
    def used(self) -> float:
        """Total time reserved so far."""
        return float(sum(self.reserved.values()))

    @property
    def available(self) -> float:
        """Time still unreserved."""
        return max(0.0, self.total - self.used)

    def deadline_for(self, name: str, *, clock: object = time.monotonic) -> Deadline:
        """Return the deadline for a reserved stage.

        The stage gets the time that is left once the *other* reserved
        stages have had theirs.

        Args:
            name: The stage name.
            clock: Monotonic clock callable.

        Returns:
            A deadline for the stage.

        Raises:
            KeyError: If the stage was never reserved.
        """
        if name not in self.reserved:
            raise KeyError(f"stage {name!r} has not been reserved")
        others = sum(amount for stage, amount in self.reserved.items() if stage != name)
        end = clock() + max(0.0, self.total - others)  # type: ignore[operator]
        return Deadline(end)
